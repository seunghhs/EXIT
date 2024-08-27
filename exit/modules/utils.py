import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error
from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

#===================== loss =====================
def compute_pv_loss(module, results, normalizer):
    logits = module.pv_head(results['cls_feats'])
    labels = torch.FloatTensor(results['pv']).to(logits.device)

     # normalize encode if config["mean"] and config["std], else pass
    logits = logits.squeeze(-1)
    labels = normalizer.encode(labels)
    loss = F.mse_loss(logits, labels)

    labels = labels.to(torch.float32)
    logits = logits.to(torch.float32)

    results =  {
        'pv_loss': loss,
        'pv_logits': normalizer.decode(logits), 
       'pv_labels':  normalizer.decode(labels),
           }

    # call update() loss and acc
    phase = "train" if module.training else "val"
    loss = getattr(module, f"{phase}_pv_loss")(results["pv_loss"])
    mae = getattr(module, f"{phase}_pv_mae")(
        mean_absolute_error(results["pv_logits"], results["pv_labels"])
    )

    if module.write_log:
        module.log(f"pv/{phase}/loss", loss, sync_dist=True)
        module.log(f"pv/{phase}/mae", mae, sync_dist=True)    

    return results




def compute_sa_loss(module, results, normalizer):
    logits = module.sa_head(results['cls_feats'])
    labels = torch.FloatTensor(results['sa']).to(logits.device)

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

    if module.write_log:
        module.log(f"sa/{phase}/loss", loss, sync_dist=True)
        module.log(f"sa/{phase}/mae", mae, sync_dist=True)    

    return results

def compute_regression_loss(module, results, normalizer):
    logits = module.regression_head(results['cls_feats'])
    labels = torch.FloatTensor(results['regression']).to(logits.device)

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

    if module.write_log:
        module.log(f"regression/{phase}/loss", loss, sync_dist=True)
        module.log(f"regression/{phase}/mae", mae, sync_dist=True)    

    return results


def compute_classification_loss(module, results):
    infer = module.infer(batch)

    logits, binary = module.classification_head(
        infer["cls_feats"]
    )  # [B, output_dim]
    labels = torch.LongTensor(results["classification"]).to(logits.device)  # [B]
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
        module.log(f"classification/{phase}/loss", loss, sync_dist=True)
        module.log(f"classification/{phase}/accuracy", acc, sync_dist=True)

    return ret
#===================== loss =====================

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