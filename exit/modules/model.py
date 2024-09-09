from typing import Any, List
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from exit.modules import heads
from exit.modules.visiontransformer import VisionTransformer1D
from exit.modules.utils import Normalizer, init_weights
from exit.modules.utils import compute_pv_loss, compute_sa_loss,compute_regression_loss, compute_classification_loss



from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class MultiModal(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()


        # set vision transformer
        self.vision_transformer = VisionTransformer1D(
        seq_length = config['model']['seq_length'],
        patch_size = config['model']['patch_size'],
        in_chans = config['model']['in_chans'],
        embed_dim = config['model']['embed_dim'],               
        )

        self.visualize = config['visualize']
        self.hidden_dim = config['model']['hidden_dim']
        self.exclude_keys = ['xrd', 'ref', 'name']
        self.current_tasks = []
        self.write_log = True
        ##################################################
        # set mofid transformer (JW)
        self.mofid_transformer = None
        ##################################################
        
        # class token
        self.cls_embeddings = nn.Linear(1, self.hidden_dim)
        self.cls_embeddings.apply(init_weights)

        # token type embedding
        self.token_type_embeddings = nn.Embedding(2, self.hidden_dim)
        self.token_type_embeddings.apply(init_weights)        

        # pooler
        self.pooler = heads.Pooler(self.hidden_dim)
        self.pooler.apply(init_weights)


        # ===================== loss =====================
        if config["loss_names"]["pv"] > 0:
            self.pv_head = heads.PVHead(self.hidden_dim)
            self.pv_head.apply(init_weights)
            self.current_tasks.append('pv')
            self.pv_mean = config['pv_mean']
            self.pv_std = config['pv_std']
            
        if config["loss_names"]["sa"] > 0:
            self.sa_head = heads.SAHead(self.hidden_dim)
            self.sa_head.apply(init_weights)
            self.current_tasks.append('sa')
            self.sa_mean = config['sa_mean']
            self.sa_std = config['sa_std']            

        if config["loss_names"]["mofid"] > 0:
            self.mofid_head = heads.MOFidHead(self.hidden_dim)
            self.mofid_head.apply(init_weights)
            self.current_tasks.append('mofid')
            
        if config["loss_names"]["regression"] > 0:
            self.regression_head = heads.RegressionHead(self.hidden_dim)
            self.regression_head.apply(init_weights)
            self.current_tasks.append('regression')
            self.regression_mean = config['regression_mean']
            self.regression_std = config['regression_std']             

        if config["loss_names"]["classification"] > 0:
            self.classification_head = heads.ClassificationHead(self.hidden_dim)
            self.classification_head.apply(init_weights)
            self.current_tasks.append('classification')


        # #===================== load pretrained model =====================
        # if config['model_path'] is not None:
        #     ckpt = torch.load(config['model_path'])

    
    def forward(self, batch):
        B = len(batch['xrd'])

        
##################################################
        # mofid transformer (JW)
        mfoid_embeds, mofid_masks, mofid_labels = None, None, None
##################################################
        
        # class tokens
        cls_tokens = torch.zeros(B).to(graph_embeds)  # [B]
        cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        cls_mask = torch.ones(B, 1).to(graph_masks)  # [B, 1]

        # class tokens + mofid_tokens
        mofid_embeds = torch.cat([cls_embeds, mofid_embeds], dim=1 )
        mofid_masks = torch.cat([cls_mask, mofid_masks], dim=1)

        
        # vision transformer (xrd)
        xrd_embeds, xrd_masks, xrd_labels = self.vision_transformer(batch['xrd'] )

        # add token_type_embedding (mofid ->0, xrd->1)
        mofid_embeds = mofid_embeds + self.token_type_embeddings(
            torch.zeros_like(mofid_masks, device=self.device).long()
        )
        xrd_embeds = xrd_embeds + self.token_type_embeddings(
            torch.ones_like(xrd_masks, device=self.device).long()
        )        

        x = torch.cat([mofid_embeds, xrd_embeds], dim=1)
        x_masks = torch.cat([mofid_masks, xrd_masks], dim=1)

        # transformer blocks
        attn_weights = []
        for i, blk in enumerate(self.vision_transformer.blocks):
            x, _attn = blk(x, mask=x_masks)

            if self.vis:
                attn_weights.append(_attn)

        x = self.vision_transformer.norm(x)
        cls_feats = self.pooler(x)

        mofid_feats, xrd_feats = (
            x[:, : mofid_embeds.shape[1]],
            x[:, mofid_embeds.shape[1] :],            
        )

        # get batch for target values
        results = {key: value for key, value in batch.items() if key not in self.exclude_keys}
        results.update({
            'cls_feats': cls_feats,
            'raw_cls_feats': x[:, 0],
            'mofid_feats': mofid_feats,
            'mofid_masks': mofid_masks,
            'xrd_feats': xrd_feats,
            'xrd_masks': xrd_masks,
            'xrd_labels': xrd_labels,
            'attn_weights': attn_weights,

        })

        # calculate losses
        loss_dict = self.get_loss(results)
        results.update(loss_dict)
        
        return results


    def get_loss(self, results ):
        losses = dict()
        if not len(self.current_tasks):
            return losses

        if 'pv' in self.current_tasks:
            normalizer = Normalizer(self.pv_mean, self.pv_std, self.device)
            losses.update(compute_pv_loss(self, results, normalizer))

        if 'sa' in self.current_tasks:
            normalizer = Normalizer(self.sa_mean, self.sa_std, self.device)
            losses.update(compute_sa_loss(self, results, normalizer))

##################################################
        # mofid loss JW
        if 'mofid' in self.current_tasks:
            pass
##################################################

        
        if 'regression' in self.current_tasks:
            normalizer = Normalizer(self.regression_mean, self.regression_std, self.device)
            losses.update(compute_regression_loss(self, results, normalizer))

        if 'classification' in self.current_tasks:
            losses.update(compute_classification_loss(self, results))

        return losses


    def training_step(self, batch, batch_idx):
        output = self(batch)
        #total_loss = 