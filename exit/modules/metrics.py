"""
Custom TorchMetrics classes for distributed training.

Accuracy: handles both multi-class (argmax) and binary (threshold 0.5) predictions.
Scalar: running average of arbitrary scalar values across all steps in an epoch.
These are attached dynamically to the Lightning module by set_metrics() in utils.py.
"""
import torch
from torchmetrics import Metric


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits = logits.detach().to(self.correct.device)
        target = target.detach().to(self.correct.device)

        if len(logits.shape) > 1:
            preds = logits.argmax(dim=-1)
        else:
            # binary accuracy
            preds = (logits >= 0.5).to(target.dtype)

        if target.numel() == 0:
            return

        assert preds.shape == target.shape

        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total