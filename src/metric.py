import torch


class Metric:
    def __init__(self, device, eval_k):
        self.device = device
        self.eval_k = eval_k
        self.dcg_weight = self.get_weight()

    def get_weight(self):
        dcg_weight = torch.reciprocal(torch.log2(torch.arange(2, self.eval_k+2)))
        return dcg_weight.to(self.device)

    def __call__(self, score, output):
        indices = self.top_k(score)

        ndcg = self.normalized_discounted_cumulative_grain(indices, output)
        precision = self.precision(indices, output)
        recall = self.recall(indices, output)
        mrr = self.mean_reciprocal_rank(indices, output)
        mrp = self.mean_rank_percentile(indices, output)

        return {
            "ndcg": ndcg,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
            "mrp": mrp,
        }

    def normalized_discounted_cumulative_grain(self, indices, output):
        """
        Args:
        Returns:
        """
        outputs = output.view(-1, 1).expand_as(indices)
        hits = (outputs == indices).float()

        dcg = torch.sum(hits * self.dcg_weight.expand_as(hits), dim=1)
        idcg = 1.0
        ndcg = dcg / idcg
        return torch.mean(ndcg).item()

    @staticmethod
    def recall(indices, output):
        """
        Args:
        Returns:
        """
        outputs = output.view(-1, 1).expand_as(indices)
        n_hits = (outputs == indices).nonzero()[:, :-1].size(0)
        recall = float(n_hits) / outputs.size(0)
        return recall

    def precision(self, indices, output):
        """
        Args:
        Returns:
        """
        outputs = output.view(-1, 1).expand_as(indices)
        hits = (outputs == indices)
        precision = torch.sum(hits, axis=1) / self.eval_k
        return torch.mean(precision).item()

    @staticmethod
    def mean_reciprocal_rank(indices, output):
        """
        Args:
        Returns:
        """
        outputs = output.view(-1, 1).expand_as(indices)
        hits = (outputs == indices).nonzero()
        ranks = hits[:, -1] + 1
        rr = torch.reciprocal(ranks)
        return torch.sum(rr).item() / output.size(0)

    @staticmethod
    def mean_rank_percentile(indices, output):
        """
        Args:
        Returns:
        """
        pass

    @staticmethod
    def coverage(indices, n_items):
        """
        Args:
        Returns:
        """
        rec_items = indices.view(-1).unique()
        return rec_items.size(0) / n_items

    def top_k(self, score):
        _, indices = torch.topk(score, self.eval_k, -1)
        return indices
