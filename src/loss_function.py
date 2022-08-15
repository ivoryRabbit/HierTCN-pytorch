import torch
from torch import nn, Tensor
from torch.nn.functional import logsigmoid


class LossFunction(nn.Module):
    def __init__(self, loss_type="TOP1"):
        super(LossFunction, self).__init__()

        if loss_type == "L2Loss":
            self._loss_fn = L2Loss()
        elif loss_type == "NCE":
            self._loss_fn = NCE()
        elif loss_type == "BPR":
            self._loss_fn = BPR()
        elif loss_type == "TOP1-HingeLoss":
            self._loss_fn = HingeLoss()
        else:
            raise NotImplementedError

    def forward(self, logit):
        return self._loss_fn(logit)


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    @staticmethod
    def forward(item: Tensor, user: Tensor) -> Tensor:
        loss = torch.square(item - user).mean()
        return loss


class NCE(nn.Module):
    def __init__(self):
        super(NCE, self).__init__()

    @staticmethod
    def forward(item: Tensor, user: Tensor):
        pass


class BPR(nn.Module):
    def __init__(self):
        super(BPR, self).__init__()

    @staticmethod
    def forward(logit: Tensor) -> Tensor:
        diff = logit.diag().view(-1, 1).expand_as(logit).T - logit
        loss = -torch.mean(logsigmoid(diff))
        return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    @staticmethod
    def forward(logit: Tensor) -> Tensor:
        pass