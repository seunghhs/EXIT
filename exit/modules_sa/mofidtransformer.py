import torch
from torch import nn, Tensor  # Import Tensor here
import math 
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce




class PositionalEncoding(nn.Module):
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MOFidEncoder(nn.Module):
    
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
        
        #positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout) 
        
        #encoder 
        # encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)

        self.d_model = d_model
        self.init_weights()
        
    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        #output = self.transformer_encoder(src)
        
        return src
    
    



