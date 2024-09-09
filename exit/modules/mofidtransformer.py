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


class BasicTransformer(nn.Module):
    
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
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)

        self.d_model = d_model
        self.init_weights()
        
    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        return output
    
    

class MaskingGenerator(nn.Module):
    def __init__(
        self,
        mask_prob=0.15,
        replace_prob=0.9,
        num_tokens=None,
        random_token_prob=0.,
        mask_token_id=14,
        pad_token_id=0,
        mask_ignore_token_ids=[]):
        super().__init__()

        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mask_ignore_token_ids = set(mask_ignore_token_ids + [pad_token_id])


    def prob_mask_like(self, t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob

    def mask_with_tokens(self, t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

    def get_mask_subset_with_prob(self, mask, prob):
        # mask.shape 에서 batch와 seq_len 추출
        batch, seq_len = mask.shape
        # 텐서가 위치한 device 정보를 직접 가져옴
        device = mask.device
        max_masked = math.ceil(prob * seq_len)

        num_tokens = mask.sum(dim=-1, keepdim=True)
        mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
        mask_excess = mask_excess[:, :max_masked]

        rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
        _, sampled_indices = rand.topk(max_masked, dim=-1)
        sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        return new_mask[:, 1:].bool()

    def forward(self, seq):
        # Mask preparation
        no_mask = self.mask_with_tokens(seq, self.mask_ignore_token_ids)
        mofid_mask = self.get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # Mask input with mask tokens with probability of `replace_prob`
        masked_seq = seq.clone().detach()

        # Derive labels for prediction
        mofid_labels = seq.masked_fill(~mofid_mask, self.pad_token_id)

        # If random token probability > 0 for MLM
        if self.random_token_prob > 0:
            random_token_prob = self.prob_mask_like(seq, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, seq.shape, device=seq.device)
            random_no_mask = self.mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)
            mofid_labels = torch.where(random_token_prob, torch.full_like(mofid_labels, self.pad_token_id), mofid_labels)  # Update labels to ignore random tokens

        # Apply mask token id
        replace_prob = self.prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mofid_mask & replace_prob, self.mask_token_id)
        
        return masked_seq, mofid_mask, mofid_labels

