from typing import Any, List
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from exit.modules import heads
from exit.modules.visiontransformer import VisionTransformer1D
from exit.modules.mofidtransformer import  MOFidEncoder
from exit.modules.utils import Normalizer, init_weights
from exit.modules.utils import compute_vf_loss, compute_sa_loss,compute_regression_loss, compute_classification_loss, compute_mofid_loss, compute_xrd_loss
from exit.modules.utils import epoch_wrapup, set_schedule, set_metrics



from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score


class MultiModal(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.ntoken = config['model']['ntoken']
        self.visualize = config['visualize']
        self.hidden_dim = config['model']['hidden_dim']
        self.exclude_keys = ['xrd', 'ref', 'name']
        self.current_tasks = []
        self.write_log = True
        self.vis = False
        self.xrd_mask = False

        
        # ===================== loss =====================
        if config["loss_names"]["vf"] > 0:
            self.vf_head = heads.VFHead(self.hidden_dim)
            self.vf_head.apply(init_weights)
            self.current_tasks.append('vf')
            self.vf_mean = config['vf_mean']
            self.vf_std = config['vf_std']


        if config["loss_names"]["xrd"] > 0:
            self.xrd_mask = True
            self.xrd_head = heads.XRDHead(self.hidden_dim, config['model']['patch_size'])
            self.xrd_head.apply(init_weights)
            self.current_tasks.append('xrd')


        if config["loss_names"]["mofid"] > 0:
            self.mofid_head = heads.MOFidHead(self.hidden_dim, self.ntoken)
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

        self.weighted_ratio = {
            task: config.get(f'{task}_weight', 1)  # Example: Get 'sa_weight' if it exists, set it to 1 if not.
            for task in self.current_tasks
        }
        # #===================== load pretrained model =====================
        # if config['model_path'] is not None:
        #     ckpt = torch.load(config['model_path'])

        self.test_logits = []
        self.test_labels = []
        self.vf_test_logits = []
        self.vf_test_labels = []
        self.xrd_test_logits = []
        self.xrd_test_labels = []

        # set vision transformer
        self.vision_transformer = VisionTransformer1D(
        seq_length = config['model']['seq_length'],
        patch_size = config['model']['patch_size'],
        in_chans = config['model']['in_chans'],
        embed_dim = config['model']['embed_dim'], 
        mask = self.xrd_mask
        )




        # mofid

        self.mofid_encoder = MOFidEncoder(ntoken = config['model']['ntoken'],
            d_model = config['model']['d_model'] , 
            nhead = config['model']['nhead'], 
            d_hid = config['model']['d_hid'],
            nlayers = config['model']['nlayers'])


        
        # class token
        self.cls_embeddings = nn.Linear(1, self.hidden_dim)
        self.cls_embeddings.apply(init_weights)

        # token type embedding
        self.token_type_embeddings = nn.Embedding(2, self.hidden_dim)
        self.token_type_embeddings.apply(init_weights)        

        # pooler
        self.pooler = heads.Pooler(self.hidden_dim)
        self.pooler.apply(init_weights)

        set_metrics(self)

    
    def forward(self, batch):
        B = len(batch['xrd'].to(self.device))
        
        #mofid encoding
        mofid_embeds,  mofid_labels = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
        mofid_attention_masks = batch['attention_mask'].to(self.device)
        mofid_masks = mofid_labels != -100
        mofid_embeds = self.mofid_encoder(mofid_embeds)
        
        
        
        # # class tokens
        # cls_tokens = torch.zeros(B).to(mofid_embeds)  # [B]
        # cls_embeds = self.cls_embeddings(cls_tokens[:, None, None])  # [B, 1, hid_dim]
        # cls_mask = torch.zeros(B, 1).to(mofid_masks)  # [B, 1]
        
        # # class tokens + mofid_tokens
        # mofid_embeds = torch.cat([cls_embeds, mofid_embeds], dim=1 )
        # mofid_masks = torch.cat([cls_mask, mofid_masks], dim=1)
        # mofid_labels = torch.cat([cls_tokens[:,None], mofid_labels], dim=1)
        
        
        # vision transformer encoding (xrd)
        xrd_embeds, xrd_labels, xrd_masks = self.vision_transformer(batch['xrd'].float().to(self.device) )
        
        # add token_type_embedding (mofid ->0, xrd->1)
        mofid_embeds = mofid_embeds + self.token_type_embeddings(
            torch.zeros_like(mofid_attention_masks, device=self.device).long()
        )
        
        xrd_embeds = xrd_embeds + self.token_type_embeddings(
            torch.ones(xrd_embeds.shape[:2], device=self.device).long()
        )        
        xrd_attention_masks = torch.ones(xrd_embeds.shape[:2], device=self.device)


        
        x = torch.cat([mofid_embeds, xrd_embeds], dim=1)
        x_masks = torch.cat([mofid_attention_masks, xrd_attention_masks], dim=1)
        
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
            #'mofid_labels': mofid_labels,
            'xrd_feats': xrd_feats,
            'xrd_masks': xrd_masks,
            'xrd_patches': xrd_labels,
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

        if 'vf' in self.current_tasks:
            vf_normalizer = Normalizer(self.vf_mean, self.vf_std, self.device)
            losses.update(compute_vf_loss(self, results, vf_normalizer))

        if 'sa' in self.current_tasks:
            sa_normalizer = Normalizer(self.sa_mean, self.sa_std, self.device)
            losses.update(compute_sa_loss(self, results, sa_normalizer))


        if 'mofid' in self.current_tasks:
            losses.update(compute_mofid_loss(self, results))

        if 'xrd' in self.current_tasks:

            losses.update(compute_xrd_loss(self, results))
        
        if 'regression' in self.current_tasks:
            normalizer = Normalizer(self.regression_mean, self.regression_std, self.device)
            losses.update(compute_regression_loss(self, results, normalizer))

        if 'classification' in self.current_tasks:
            losses.update(compute_classification_loss(self, results))

        return losses


    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss_dict = self.get_loss(output)
        total_loss = sum([
            v * self.weighted_ratio.get(k.split('_')[0], 1)  
            for k, v in loss_dict.items() 
            if "loss" in k
        ])
        self.log('train_loss', total_loss, sync_dist=True, logger=True, prog_bar=True )
        

        
        return total_loss

    def on_train_epoch_end(self):
        epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss_dict = self.get_loss(output)
        total_loss = sum([
            v * self.weighted_ratio.get(k.split('_')[0], 1)  
            for k, v in loss_dict.items() 
            if "loss" in k
        ])
        return total_loss
         
    def on_validation_epoch_end(self) -> None:
        epoch_wrapup(self) 


    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss_dict = self.get_loss(output)
        
        output = {
            k: (v.cpu() if torch.is_tensor(v) else v) for k, v in loss_dict.items()
        }  # update cpu for memory

        if "regression_logits" in output.keys():
            self.test_logits += output["regression_logits"].tolist()
            self.test_labels += output["regression_labels"].tolist()

        if "vf_logits" in output.keys():
            self.vf_test_logits += output["vf_logits"].tolist()
            self.vf_test_labels += output["vf_labels"].tolist()
            
        # if "sa_logits" in output.keys():
        #     self.sa_test_logits += output["sa_logits"].tolist()
        #     self.sa_test_labels += output["sa_labels"].tolist()

        if "xrd_logits" in output.keys():
            self.xrd_test_logits += output["xrd_logits"].tolist()
            self.xrd_test_labels += output["xrd_labels"].tolist()            
            
        
            
        return output

    def on_test_epoch_end(self):
        epoch_wrapup(self)

        # calculate r2 score when regression
        if len(self.test_logits) > 1:
            r2 = r2_score(np.array(self.test_labels), np.array(self.test_logits))
            mae = mean_absolute_error(np.array(self.test_labels), np.array(self.test_logits))
            self.log(f"test/r2_score", r2, sync_dist=True)
            self.log(f"test/mae", mae, sync_dist=True )
            self.test_labels.clear()
            self.test_logits.clear()

        if len(self.vf_test_logits) > 1:
            r2 = r2_score(np.array(self.vf_test_labels), np.array(self.vf_test_logits))
            mae = mean_absolute_error(np.array(self.vf_test_labels), np.array(self.vf_test_logits))
            self.log(f"test/vf_r2_score", r2, sync_dist=True)
            self.log(f"test/vf_mae", mae, sync_dist=True )

            self.vf_test_labels.clear()
            self.vf_test_logits.clear()
            

        if len(self.xrd_test_logits) > 1:

            r2 = r2_score(np.array(self.xrd_test_labels), np.array(self.xrd_test_logits))
            mae = mean_absolute_error(np.array(self.xrd_test_labels), np.array(self.xrd_test_logits))
            self.log(f"test/xrd_r2_score", r2, sync_dist=True)
            self.log(f"test/xrd_mae", mae, sync_dist=True )
        
            self.xrd_test_labels.clear()
            self.xrd_test_logits.clear()
            
    
    def configure_optimizers(self):
        return set_schedule(self)


    def on_predict_start(self):
        self.write_log = False


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)

        if "classification_logits" in output:
            if self.hparams.config["n_classes"] == 2:
                output["classification_logits_index"] = torch.round(
                    output["classification_logits"]
                ).to(torch.int)
            else:
                softmax = torch.nn.Softmax(dim=1)
                output["classification_logits"] = softmax(
                    output["classification_logits"]
                )
                output["classification_logits_index"] = torch.argmax(
                    output["classification_logits"], dim=1
                )

        output = {
            k: (v.cpu().tolist() if torch.is_tensor(v) else v)
            for k, v in output.items()
            if ("logits" in k) or ("labels" in k) 
        }

        return output

    def on_predict_epoch_end(self, *args):
        self.test_labels.clear()
        self.test_logits.clear()

    def on_predict_end(
        self,
    ):
        self.write_log = True

    def lr_scheduler_step(self, scheduler, *args):
        #print(f"Calling scheduler.step() at epoch {self.current_epoch}, step {self.global_step}")
        if len(args) == 2:
            optimizer_idx, metric = args
        elif len(args) == 1:
            (metric,) = args
        else:
            raise ValueError(
                "lr_scheduler_step must have metric and optimizer_idx(optional)"
            )

        if pl.__version__ >= "2.0.0":
            scheduler.step(epoch=self.current_epoch)
        else:
            scheduler.step()