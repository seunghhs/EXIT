"""
MOFid encoder: token embedding + sinusoidal positional encoding.

Note: The TransformerEncoder layers are intentionally not used here. MOFid tokens
are embedded and positionally encoded, then processed jointly with XRD patch tokens
by the shared transformer blocks in VisionTransformer1D (see MultiModal.forward).
"""
import torch
from torch import nn, Tensor
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from functools import reduce


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: [seq_len, batch_size, d_model]"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MOFidEncoder(nn.Module):
    """
    Encodes tokenized MOFid strings into continuous embeddings.

    Pipeline: token IDs → Embedding (scaled by sqrt(d_model)) → sinusoidal pos encoding.
    The TransformerEncoder is commented out by design: self-attention over MOFid tokens
    is handled later in the shared transformer blocks together with XRD tokens.
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
        mask_prob: float = 0.15,
        replace_prob: float = 0.9,
    ):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # TransformerEncoder intentionally disabled — shared blocks in VisionTransformer1D
        # handle cross-modal attention for both MOFid and XRD sequences.
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return src
    
    



