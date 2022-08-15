from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class TemporalConv1d(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        dilation_rate: int = 2,
        dropout_rate: float = 0.0,  # the paper does not recommend dropout CNN units
     ):
        super(TemporalConv1d, self).__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.num_layers = self._calculate_proper_num_layers()

        layers = []
        for idx in range(self.num_layers):
            dilation = self.dilation_rate ** idx
            pad_size = self._calculate_pad_size(dilation)

            temporal_block = nn.Sequential(
                nn.ConstantPad1d((pad_size, 0), 0),
                nn.Conv1d(self.input_dim, self.output_dim, self.kernel_size, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Conv1d(self.input_dim, self.output_dim, 1),  # batch normalization
            )
            layers.append(temporal_block)

        self.module_list = nn.ModuleList(layers)

    def _calculate_proper_num_layers(self) -> int:
        """
        Note that the receptive field size of an upper layer unit  = dilation * (kernel_size - 1) + 1

        Find minimal num_layers such that:
            sequence_length <= (1 + dilation + dilation^2 + ... + dilation^num_layers) * (kernel_size - 1) + 1
        <=> sequence_length - 1 <= (dilation^num_layers - 1) / (dilation - 1) * (kernel_size - 1)
        <=> (sequence_length - 1) * (dilation - 1) <= (dilation^num_layers - 1) * (kernel_size - 1)
        """
        s = self.sequence_length
        d = self.dilation_rate
        k = self.kernel_size

        num_layers = 1
        while (s - 1) * (d - 1) > (d ** num_layers - 1) * (k - 1):
            num_layers += 1

        return num_layers

    def _calculate_pad_size(self, dilation: int) -> int:
        """
        Note that output_sequence_length = input_sequence_length - dilation * (kernel_size - 1)
        Hence, we can apply forward padding (dilation * (kernel_size - 1)) zeros for same length
        """
        return dilation * (self.kernel_size - 1)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        for layer in self.module_list:
            outputs = layer(outputs) + outputs
        return outputs


class GRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.0, bias: bool = True):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.bias = bias
        self.weight_for_inputs = nn.Linear(input_dim, 3 * hidden_dim, bias=bias)
        self.weight_for_hidden = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)

    def forward(self, inputs: Tensor, hidden: Tensor) -> Tensor:
        gate_x = self.weight_for_inputs(inputs).squeeze()
        gate_h = self.weight_for_hidden(hidden).squeeze()

        i_r, i_u, i_n = gate_x.chunk(3, 1)
        h_r, h_u, h_n = gate_h.chunk(3, 1)

        reset_gate = F.sigmoid(i_r + h_r)
        update_gate = F.sigmoid(i_u + h_u)
        new_gate = F.tanh(i_n + reset_gate * h_n)

        dropout_new_gate = F.dropout(new_gate, self.dropout_rate, self.training)

        hidden_updated = dropout_new_gate + update_gate * (hidden - reset_gate)
        return hidden_updated


class HierTCN(nn.Module):
    def __init__(
        self,
        device,
        input_size,
        output_size,
        hidden_dim,
        dropout_rate=0.0,
    ):
        super(HierTCN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # layers
        self.user_to_session = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout_init)
        )
        self.session_to_output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_size),
            nn.Tanh(),
        )

        self.session_tcn = TemporalConv1d(self.input_size, self.hidden_dim, self.hidden_dim)
        self.user_gru = GRUCell(self.hidden_dim, self.hidden_dim, self.dropout_rate)
        self.to(device)

    def forward(
        self,
        inputs: Tensor,
        session_start_token: Tensor,
        user_state: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        item_embeddings = self.one_hot_encode(inputs)
        user_state_updated = self.user_gru(item_embeddings.mean(dim=-1), user_state)  # Equation 10

        # session_inputs = torch.roll(item_embeddings, 1, -1)
        # session_inputs[:, :, 0] = 0.0
        session_inputs = torch.cat([session_start_token, item_embeddings[:, :, :-1]], dim=-1)
        session_inputs = torch.cat([session_inputs, user_state.unsqueeze(-1)], dim=1)
        outputs = self.session_tcn(session_inputs)

        return outputs, user_state_updated

    def one_hot_encode(self, inputs: Tensor) -> Tensor:
        encoded = F.one_hot(inputs, num_classes=self.input_size).float()
        return encoded.to(self.device)

    def init_user_state(self, batch_size):
        hidden = torch.zeros(batch_size, self.hidden_dim)
        return hidden.to(self.device)

    def init_model(self, sigma):
        for p in self.parameters():
            if sigma > 0:
                p.data.uniform_(-sigma, sigma)
            elif len(p.size()) > 1:
                sigma_ = (6.0 / (p.size(0) + p.size(1))) ** 0.5
                if sigma == -1:
                    p.data.uniform_(-sigma_, sigma_)
                else:
                    p.data.uniform_(0, sigma_)

    def save(self, save_dir):
        pass