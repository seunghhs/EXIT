from typing import Any, List
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from exit.modules import heads
from exit.modules.visiontransformer import VisionTransformer1D
from exit.modules.utils import Normalizer, init_weights


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
        
        # set mofid transformer
        self.mofid_transformer = None
        
        # class token
        self.cls_embeddings = nn.Linear(1, config['model']["hidden_dim"])
        self.cls_embeddings.apply(init_weights)

        # pooler
        self.pooler = heads.Pooler(config['model']["hidden_dim"])
        self.pooler.apply(init_weights)


        # ===================== loss =====================
        if config["loss_names"]["pv"] > 0:
            self.pv_head = heads.PVHead(config["hidden_dim"])
            self.pv_head.apply(init_weights)
            
        if config["loss_names"]["sa"] > 0:
            self.sa_head = heads.SAHead(config["hidden_dim"])
            self.sa_head.apply(init_weights)

        if config["loss_names"]["mofid"] > 0:
            self.mofid_head = heads.MOFidHead(config["hidden_dim"])
            self.mofid_head.apply(init_weights)
            
        if config["loss_names"]["regression"] > 0:
            self.regression_head = heads.RegressionHead(config["hidden_dim"])
            self.regression_head.apply(init_weights)

        if config["loss_names"]["classification"] > 0:
            self.classification_head = heads.ClassificationHead(config["hidden_dim"])
            self.classification_head.apply(init_weights)

