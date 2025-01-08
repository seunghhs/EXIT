import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error, r2_score
from torch.optim import AdamW
import pytorch_lightning as pl
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)
from exit.modules.metrics import Accuracy, Scalar

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k in ['regression', 'vf', 'xrd' ]:
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_mae", Scalar())
                setattr(pl_module, f"{split}_{k}_r2", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

                
#===================== loss =====================
def compute_vf_loss(module, results, normalizer):
    logits = module.vf_head(results['cls_feats'])
    labels = (results['vf']).to(logits.device)

     # normalize encode if config["mean"] and config["std], else pass
    logits = logits.squeeze(-1)
    labels = normalizer.encode(labels)
    loss = F.mse_loss(logits, labels)

    labels = labels.to(torch.float32)
    logits = logits.to(torch.float32)

    results =  {
        'vf_loss': loss,
        'vf_logits': normalizer.decode(logits), 
       'vf_labels':  normalizer.decode(labels),
           }

    # call update() loss and acc
    phase = "train" if module.training else "val"
    loss = getattr(module, f"{phase}_vf_loss")(results["vf_loss"])
    mae = getattr(module, f"{phase}_vf_mae")(
        mean_absolute_error(results["vf_logits"], results["vf_labels"])
    )
    
    r2 = getattr(module, f"{phase}_vf_r2")(
        r2_score(results["vf_logits"], results["vf_labels"])
    )

    if module.write_log:
        module.log(f"vf/{phase}/loss", loss, on_step=False, on_epoch=True,sync_dist=True)
        module.log(f"vf/{phase}/mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        module.log(f"vf/{phase}/r2", r2, on_step=False, on_epoch=True, sync_dist=True)

    return results



def compute_sa_loss(module, results, normalizer):
    logits = module.sa_head(results['cls_feats'])
    labels = (results['sa']).to(logits.device)

     # normalize encode if config["mean"] and config["std], else pass
    logits = logits.squeeze(-1)
    labels = normalizer.encode(labels)
    loss = F.mse_loss(logits, labels)

    labels = labels.to(torch.float32)
    logits = logits.to(torch.float32)

    results =  {
        'sa_loss': loss,
        'sa_logits': normalizer.decode(logits), 
       'sa_labels':  normalizer.decode(labels),
           }

    # call update() loss and acc
    phase = "train" if module.training else "val"
    loss = getattr(module, f"{phase}_sa_loss")(results["sa_loss"])
    mae = getattr(module, f"{phase}_sa_mae")(
        mean_absolute_error(results["sa_logits"], results["sa_labels"])
    )
    r2 = getattr(module, f"{phase}_sa_r2")(
        r2_score(results["sa_logits"], results["sa_labels"])
    )

    if module.write_log:
        module.log(f"sa/{phase}/loss", loss, on_step=False,  on_epoch=True, sync_dist=True)
        module.log(f"sa/{phase}/mae", mae, on_step=False, on_epoch=True, sync_dist=True) 
        module.log(f"sa/{phase}/r2", r2, on_step=False, on_epoch=True, sync_dist=True) 

    return results




def compute_mofid_loss(module, results):


    logits = module.mofid_head(
        results["mofid_feats"]
    )  # [B, output_dim]
    
    masks = (results['mofid_masks']).to(logits.device)
    labels = (results["labels"]).to(logits.device)  # [B]
    

    loss = F.cross_entropy(logits[masks], labels[masks])

    results = {
        "mofid_loss": loss,
        "mofid_logits": logits[masks],
        "mofid_labels": labels[masks],
    }

    # call update() loss and acc
    phase = "train" if module.training else "val"
    loss = getattr(module, f"{phase}_mofid_loss")(
        results["mofid_loss"]
    )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    acc = getattr(module, f"{phase}_mofid_accuracy")(
        results["mofid_logits"], results["mofid_labels"]
    )

    if module.write_log:
        module.log(f"mofid/{phase}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        module.log(f"mofid/{phase}/accuracy", acc, on_step=False, on_epoch=True, sync_dist=True)

    return results




def compute_xrd_loss(module, results, ):

    logits = module.xrd_head(results['xrd_feats'][:,1:,:])
    labels = (results['xrd_patches']).to(logits.device)
    masks = (results["xrd_masks"]==-100).to(logits.device)
    
    labels = labels[masks]
    logits = logits[masks]
    
    loss = F.mse_loss(logits, labels)
    
    labels = labels.to(torch.float32)
    logits = logits.to(torch.float32)

    results =  {
        'xrd_loss': loss,
        'xrd_logits': logits, 
       'xrd_labels':  labels,
           }

    # call update() loss and acc
    phase = "train" if module.training else "val"
    loss = getattr(module, f"{phase}_xrd_loss")(results["xrd_loss"])
    mae = getattr(module, f"{phase}_xrd_mae")(
        mean_absolute_error(results["xrd_logits"], results["xrd_labels"])
    )
    r2 = getattr(module, f"{phase}_xrd_r2")(
        r2_score(results["xrd_logits"], results["xrd_labels"])
    )

    if module.write_log:
        module.log(f"xrd/{phase}/loss", loss, on_step=False,  on_epoch=True, sync_dist=True)
        module.log(f"xrd/{phase}/mae", mae, on_step=False, on_epoch=True, sync_dist=True) 
        module.log(f"xrd/{phase}/r2", r2, on_step=False, on_epoch=True, sync_dist=True) 


    return results



def compute_regression_loss(module, results, normalizer):
    logits = module.regression_head(results['cls_feats'])
    labels = (results['regression']).to(logits.device)

     # normalize encode if config["mean"] and config["std], else pass
    logits = logits.squeeze(-1)
    labels = normalizer.encode(labels)
    loss = F.mse_loss(logits, labels)

    labels = labels.to(torch.float32)
    logits = logits.to(torch.float32)

    results =  {
        'regression_loss': loss,
        'regression_logits': normalizer.decode(logits), 
       'regression_labels':  normalizer.decode(labels),
           }

    # call update() loss and acc
    phase = "train" if module.training else "val"
    loss = getattr(module, f"{phase}_regression_loss")(results["regression_loss"])
    mae = getattr(module, f"{phase}_regression_mae")(
        mean_absolute_error(results["regression_logits"], results["regression_labels"])
    )
    # r2 = getattr(module, f"{phase}_regression_r2")(
    #     r2_score(results["regression_logits"], results["regression_labels"])
    # )

    if module.write_log:
        module.log(f"regression/{phase}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        module.log(f"regression/{phase}/mae", mae, on_step=False, on_epoch=True, sync_dist=True) 
        #module.log(f"regression/{phase}/r2", r2, on_step=False, on_epoch=True,sync_dist=True) 

    return results


def compute_classification_loss(module, results):

    logits, binary = module.classification_head(
        results["cls_feats"]
    )  # [B, output_dim]
    labels = (results["classification"]).to(logits.device)  # [B]
    assert len(labels.shape) == 1
    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float())
    else:
        loss = F.cross_entropy(logits, labels)

    results = {
        "classification_loss": loss,
        "classification_logits": logits,
        "classification_labels": labels,
    }

    # call update() loss and acc
    phase = "train" if module.training else "val"
    loss = getattr(module, f"{phase}_classification_loss")(
        results["classification_loss"]
    )                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    acc = getattr(module, f"{phase}_classification_accuracy")(
        results["classification_logits"], results["classification_labels"]
    )

    if module.write_log:
        module.log(f"classification/{phase}/loss", loss, on_step=False, on_epoch=True,sync_dist=True)
        module.log(f"classification/{phase}/accuracy", acc, on_step=False, on_epoch=True, sync_dist=True)

    return ret
#===================== loss =====================

# def epoch_wrapup(pl_module):
#     phase = "train" if pl_module.training else "val"

#     the_metric = 0

#     for loss_name, v in pl_module.hparams.config["loss_names"].items():
#         if v < 1:
#             continue

#         if loss_name in ["regression" , 'pv', 'sa', ]:
#             # mse loss
#             pl_module.log(
#                 f"{loss_name}/{phase}/loss_epoch",
#                 getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
#                 batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
#                 sync_dist=True,
#             )
#             getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
#             # mae loss
#             value = getattr(pl_module, f"{phase}_{loss_name}_mae").compute()
#             pl_module.log(
#                 f"{loss_name}/{phase}/mae_epoch",
#                 value,
#                 batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
#                 sync_dist=True,
#             )
#             getattr(pl_module, f"{phase}_{loss_name}_mae").reset()

#             value = -value
#         else:
#             value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
#             pl_module.log(
#                 f"{loss_name}/{phase}/accuracy_epoch",
#                 value,
#                 batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
#                 sync_dist=True,
#             )
#             getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
#             pl_module.log(
#                 f"{loss_name}/{phase}/loss_epoch",
#                 getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
#                 batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
#                 sync_dist=True,
#             )
#             getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

#         the_metric += value

#     pl_module.log(f"{phase}/the_metric", the_metric, sync_dist=True)


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"

    the_metric = 0
    the_metric_2 = 0
    
    if len(pl_module.trainer.optimizers) > 0:
        for i, param_group in enumerate(pl_module.trainer.optimizers[0].param_groups):
            current_lr = param_group['lr']
            pl_module.log(f'lr_group_{i}', current_lr,   sync_dist=True)
    
    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        if loss_name in ["regression" , 'vf', 'xrd']:
            # mse 
            tmp_loss = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                tmp_loss,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            
            # mae loss
            value = getattr(pl_module, f"{phase}_{loss_name}_mae").compute()
            pl_module.log(
                f"{loss_name}/{phase}/mae_epoch",
                value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_mae").reset()


            value = -value


            r2 = getattr(pl_module, f"{phase}_{loss_name}_r2").compute()
            pl_module.log(
                f"{loss_name}/{phase}/r2_epoch",
                r2,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_r2").reset()

        
        else:
            
            # mse 
            tmp_loss = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()

            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(
                f"{loss_name}/{phase}/accuracy_epoch",
                value,
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
                batch_size=pl_module.hparams["config"]["per_gpu_batchsize"],
                sync_dist=True,
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += tmp_loss * pl_module.weighted_ratio.get(loss_name, 1)
        the_metric_2 += value * pl_module.weighted_ratio.get(loss_name, 1)

    pl_module.log(f"{phase}/the_metric", the_metric, sync_dist=True)
    pl_module.log(f"{phase}/the_metric_2", the_metric_2, sync_dist=True)


def set_schedule(module):
    lr = module.hparams.config["learning_rate"]
    wd = module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["regression_head", "classification_head"]
    lr_mult = module.hparams.config["lr_mult"]
    end_lr = module.hparams.config["end_lr"]
    decay_power = module.hparams.config["decay_power"]
    optim_type = module.hparams.config["optim_type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in module.named_parameters()
                if any(nd in n for nd in no_decay)  # within no_decay
                and not any(bb in n for bb in head_names)  # not within head_names
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in module.named_parameters()
                if not any(nd in n for nd in no_decay)  # not within no_decay
                and any(bb in n for bb in head_names)  # within head_names
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                # within no_decay and head_names
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if module.trainer.max_steps == -1:
        max_steps = module.trainer.estimated_stepping_batches
    else:
        max_steps = module.trainer.max_steps

    warmup_steps = module.hparams.config["warmup_steps"]
    if isinstance(module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    print(
        f"max_epochs: {module.trainer.max_epochs} | max_steps: {max_steps} | warmup_steps : {warmup_steps} "
        f"| weight_decay : {wd} | decay_power : {decay_power}"
    )

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    elif decay_power == "constant":
        scheduler = get_constant_schedule(
            optimizer,
        )
    elif decay_power == "constant_with_warmup":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )
        
    if pl.__version__ >= "2.0.0":
        sched = {
            "scheduler": scheduler,
            "interval": "epoch",  # epoch
            "frequency": 1,      
        }
    else: # step
        sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


class Normalizer(object):
    """
    normalize for regression
    """

    def __init__(self, mean, std, device):
        if mean and std:
            if isinstance(mean, list):
                mean = torch.tensor(mean).to(device)
            if isinstance(std, list):
                std = torch.tensor(std).to(device)
            self.mean = mean
            self.std = std
            self._norm_func = lambda tensor: (tensor - mean) / std
            self._denorm_func = lambda tensor: tensor * std + mean
        else:
            self._norm_func = lambda tensor: tensor
            self._denorm_func = lambda tensor: tensor

    def encode(self, tensor):
        return self._norm_func(tensor)

    def decode(self, tensor):
        return self._denorm_func(tensor)
    
    



###################################
#############masking###############
###################################
def prob_mask_like(t, prob): #t: tensor
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids): #t: tensor
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
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

