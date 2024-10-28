
""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer


Acknowledgments:

* Vision Transformer code from MOFTransformer
* The official code is available at https://github.com/hspark1212/MOFTransformer/blob/master/moftransformer/modules/vision_transformer_3d.py
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from torch.nn import AvgPool1d, AvgPool3d
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim, 
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        assert C % self.num_heads == 0
        qkv = (
            self.qkv(x)  # [B, N, 3*C]
            .reshape(
                B, N, 3, self.num_heads, C // self.num_heads
            )  # [B, N, 3, num_heads, C//num_heads]
            .permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, C//num_heads]
        )
        q, k, v = (
            qkv[0],  # [B, num_heads, N, C//num_heads]
            qkv[1],  # [B, num_heads, N, C//num_heads]
            qkv[2],  # [B, num_heads, N, C//num_heads]
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)  # [B, num_heads, N, N]
        attn = self.attn_drop(attn)

        x = (
            (attn @ v).transpose(1, 2).reshape(B, N, C)
        )  # [B, num_heads, N, C//num_heads] -> [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn



class PatchEmbed1D(nn.Module):
    """XRD to Patch Embedding for 1D"""

    def __init__(
        self,
        seq_length,  # sequence length  np.arange(5,50,0.01) -> 4500
        patch_size,  # length of each patch ex. 10 or 20
        in_chans=1,  # number of input channels (default 1 for 1D data)
        embed_dim=768,  # dimension of the embedding space
        no_patch_embed_bias=False,
    ):
        super().__init__()

        assert seq_length % patch_size == 0 #Sequence length must be divisible by patch size
        num_patches = seq_length // patch_size
        self.seq_length = seq_length  # sequence length ex. 4500
        self.patch_size = patch_size  # patch size ex.10 or 20
        self.num_patches = num_patches  

        self.proj = nn.Sequential(
            Rearrange(
                "b c (l p) -> b l (p c)",  
                p=patch_size
            ),
            nn.Linear(patch_size * in_chans, embed_dim) 
        )

    def forward(self, x):
        x = self.proj(x)  # [B, num_patches, embed_dim]
        return x  # output: [B, num_patches, embed_dim]



class VisionTransformer1D(nn.Module):
    """ Vision Transformer for 1D Data
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929    
    """

    def __init__(
        self,
        seq_length,
        patch_size,
        in_chans,
        embed_dim,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        add_norm_before_transformer=False,
        mpp_ratio=0.15,
    ):
        
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """     
        
        super().__init__()

        self.in_chans = in_chans
        self.mpp_ratio = mpp_ratio

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.add_norm_before_transformer = add_norm_before_transformer

        self.patch_embed = PatchEmbed1D(
            seq_length=seq_length,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.patch_size = patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if add_norm_before_transformer:
            self.pre_norm = norm_layer(embed_dim)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask_tokens(self, orig_seq, feats, patch_size, mpp_ratio):
        """
        Prepare masked tokens inputs/labels for masked patch prediction: 80% MASK, 10% random, 10% original.
        """
        avgpool = AvgPool1d(patch_size)
        with torch.no_grad():
            seq_patch = avgpool(orig_seq)
        
        labels = seq_patch.permute(0, 2, 1).contiguous()
        
        
        
        probability_matrix = torch.full(labels.shape[:-1], mpp_ratio)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100 

        return feats, labels

    def forward(self, x, mask_it=False):
        B, _, _ = x.shape
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        if mask_it:
            x, label = self.mask_tokens(x, x, self.patch_size, self.mpp_ratio)
            label = torch.cat(
                [torch.full((label.shape[0], 1, self.in_chans), -100).to(label), label],
                dim=1,
            )

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embed
        x = self.pos_drop(x)

        if self.add_norm_before_transformer:
            x = self.pre_norm(x)

        x_mask = torch.ones(x.shape[:2]).to(x)  # [B, ph*pw*pd]

        if mask_it:
            return x, x_mask, label
        else:
            return x, x_mask, None


