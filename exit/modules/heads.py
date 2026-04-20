"""
Task-specific prediction heads for EXIT.

All heads receive features from the shared transformer output and produce
task-specific logits. Active heads are instantiated dynamically in MultiModal.__init__
based on the loss_names config.
"""
import torch.nn as nn

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertPredictionHeadTransform,
)


class Pooler(nn.Module):
    """Extracts a single token (default: index 0 = CLS) and projects it through Linear+Tanh."""

    def __init__(self, hidden_size, index=0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.index = index

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, self.index]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class VFHead(nn.Module):
    """Head for void fraction regression. Activated when loss_names.vf > 0."""

    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)

    def forward(self, x):
        x = self.fc(x)
        return x


class SAHead(nn.Module):
    """Head for surface area regression. Activated when loss_names.sa > 0."""

    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)

    def forward(self, x):
        x = self.fc(x)
        return x


class MOFidHead(nn.Module):
    """
    Head for MOFid masked language modeling (MLM).

    Uses BERT's BertPredictionHeadTransform (Linear → GELU → LayerNorm) before
    the final projection to vocabulary logits. Applied only at masked token positions.
    """
    def __init__(self, hid_dim, ntoken):
        super().__init__()
        bert_config = BertConfig(hidden_size=hid_dim)
        self.transform = BertPredictionHeadTransform(bert_config)
        self.decoder = nn.Linear(hid_dim, ntoken)

    def forward(self, x):  # x: [B, max_len, hid_dim]
        x = self.transform(x)  # [B, max_len, hid_dim]
        x = self.decoder(x)    # [B, max_len, ntoken]
        return x


class RegressionHeadExp(nn.Module):
    """
    2-layer MLP regression head (hid_dim → hid_dim//2 → 1).
    Used when config['exp']=True; adds capacity over the linear RegressionHead.
    """

    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim // 2, n_targets)
        )

    def forward(self, x):
        return self.fc(x)


class RegressionHead(nn.Module):
    """Single linear regression head. Default head for SA/PV prediction."""

    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)

    def forward(self, x):
        return self.fc(x)


class ClassificationHead(nn.Module):
    """
    Classification head supporting binary (BCE) and multi-class (CE) tasks.
    Returns (logits, binary_flag) so the loss function knows which criterion to use.
    """

    def __init__(self, hid_dim, n_classes=2):
        super().__init__()
        if n_classes == 2:
            self.fc = nn.Linear(hid_dim, 1)
            self.binary = True
        else:
            self.fc = nn.Linear(hid_dim, n_classes)
            self.binary = False

    def forward(self, x):
        return self.fc(x), self.binary


class XRDHead(nn.Module):
    """
    XRD patch reconstruction head.
    Projects each patch token back to the original patch values (size = patch_size).
    Used as a self-supervised pretraining objective alongside MLM.
    """

    def __init__(self, hid_dim, n_target):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_target)

    def forward(self, x):
        return self.fc(x)



