import torch.nn as nn

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertPredictionHeadTransform,
)


class Pooler(nn.Module):
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
    """
    head for Void Fraction
    """

    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)

    def forward(self, x):
        x = self.fc(x)
        return x


class SAHead(nn.Module):
    """
    head for Surface Area
    """

    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)

    def forward(self, x):
        x = self.fc(x)
        return x


class MOFidHead(nn.Module):
    """
    head for MOFid (Masked Patch Prediction)
    """
    def __init__(self, hid_dim, ntoken):
        super().__init__()

        bert_config = BertConfig(
            hidden_size=hid_dim,
        )
        self.transform = BertPredictionHeadTransform(bert_config)
        self.decoder = nn.Linear(hid_dim, ntoken )  # bins

    def forward(self, x):  # [B, max_len, hid_dim]
        x = self.transform(x)  # [B, max_len, hid_dim]
        x = self.decoder(x)  # [B, max_len, bins]
        return x


class RegressionHeadExp(nn.Module):
    """
    head for Regression Experiments
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
    """
    head for Regression
    """

    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)

    def forward(self, x):
        x = self.fc(x)
        return x

class ClassificationHead(nn.Module):
    """
    head for Classification
    """

    def __init__(self, hid_dim, n_classes):
        super().__init__()

        if n_classes == 2:
            self.fc = nn.Linear(hid_dim, 1)
            self.binary = True
        else:
            self.fc = nn.Linear(hid_dim, n_classes)
            self.binary = False

    def forward(self, x):
        x = self.fc(x)

        return x, self.binary


class XRDHead(nn.Module):
    """
    head for XRD
    """

    def __init__(self, hid_dim, n_target):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_target)
        


    def forward(self, x):
        x = self.fc(x)

        return x


class CellHead(nn.Module):
    """
    head for Cell Parameters
    """

    def __init__(self, hid_dim, n_targets=6):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)
        


    def forward(self, x):
        x = self.fc(x)

        return x

