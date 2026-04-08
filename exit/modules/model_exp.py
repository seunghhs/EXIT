import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import r2_score, mean_absolute_error

from exit.modules.model import MultiModal


# =========================
# utils
# =========================
def masked_mean_pooling(feats, masks=None):
    """
    feats: [B, L, D]
    masks: [B, L] with 1 for valid, 0 for padding
    """
    if masks is None:
        return feats.mean(dim=1)

    masks = masks.float().unsqueeze(-1)  # [B, L, 1]
    feats = feats * masks
    denom = masks.sum(dim=1).clamp(min=1e-6)
    pooled = feats.sum(dim=1) / denom
    return pooled


class IdentityNormalizer:
    def encode(self, x):
        return x

    def decode(self, x):
        return x


class StandardNormalizer:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)

    def encode(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def decode(self, x):
        return x * (self.std + 1e-8) + self.mean


class Log1pNormalizer:
    def encode(self, x):
        return torch.log1p(x)

    def decode(self, x):
        return torch.expm1(x)


# =========================
# original-style pooler/head
# =========================
class PoolerExp(nn.Module):
    """
    first-token pooling + linear + tanh
    """
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


class RegressionHeadExp(nn.Module):
    """
    original regression head
    """
    def __init__(self, hid_dim, n_targets=1):
        super().__init__()
        self.fc = nn.Linear(hid_dim, n_targets)

    def forward(self, x):
        return self.fc(x)


# =========================
# finetune module
# =========================
class MultiModalRegressorExp(pl.LightningModule):
    def __init__(
        self,
        backbone_config,
        backbone_ckpt_path,
        lr=1e-4,
        weight_decay=1e-2,
        loss_type="mse",               # mse | huber | smoothl1 | mae
        huber_delta=1.0,
        target_transform="standard",   # standard | log1p | none
        freeze_backbone=False,
        unfreeze_last_xrd_block=False,
        train_pooler=True,
        train_token_type=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_config"])

        cfg = copy.deepcopy(backbone_config)

        self.backbone = MultiModal.load_from_checkpoint(
            backbone_ckpt_path,
            config=cfg,
            strict=False,
        )

        self.name = cfg.get("name", "exp")
        hidden_dim = self.backbone.hidden_dim

        # config options
        self.pooling_mode = cfg.get("exp_pooling_mode", "first")  # first | mean

        # pooler/head
        self.pooler = PoolerExp(hidden_dim, index=0)
        self.regression_head = RegressionHeadExp(hidden_dim, n_targets=1)

        # target normalizer
        if target_transform == "standard":
            self.normalizer = StandardNormalizer(
                mean=cfg["regression_mean"],
                std=cfg["regression_std"],
            )
        elif target_transform == "log1p":
            self.normalizer = Log1pNormalizer()
        elif target_transform == "none":
            self.normalizer = IdentityNormalizer()
        else:
            raise ValueError(f"Unknown target_transform: {target_transform}")

        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.lr = lr
        self.weight_decay = weight_decay

        # freeze strategy
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if train_token_type and hasattr(self.backbone, "token_type_embeddings"):
            for p in self.backbone.token_type_embeddings.parameters():
                p.requires_grad = True

        if unfreeze_last_xrd_block:
            for p in self.backbone.vision_transformer.blocks[-1].parameters():
                p.requires_grad = True

        # first-token pooler를 쓰는 경우에만 pooler train 의미 있음
        if train_pooler and self.pooling_mode == "first":
            for p in self.pooler.parameters():
                p.requires_grad = True
        else:
            for p in self.pooler.parameters():
                p.requires_grad = False

        for p in self.regression_head.parameters():
            p.requires_grad = True

        self.validation_step_outputs = []
        self.test_step_outputs = []

    # -------------------------
    # pooling
    # -------------------------
    def pool_with_mode(self, feats, masks=None, mode="first"):
        if mode == "mean":
            return masked_mean_pooling(feats, masks)
        elif mode == "first":
            return self.pooler(feats)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    # -------------------------
    # feature extraction
    # -------------------------
    def extract_pooled_feats(self, batch):
        backbone = self.backbone
        device = self.device

        # -------------------------
        # MOFid encoding
        # -------------------------
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        mofid_embeds = backbone.mofid_encoder(input_ids)
        mofid_embeds = mofid_embeds + backbone.token_type_embeddings(
            torch.zeros_like(attention_mask, device=device).long()
        )

        # -------------------------
        # XRD encoding
        # -------------------------
        xrd_embeds, _, _ = backbone.vision_transformer(
            batch["xrd"].float().to(device)
        )

        xrd_embeds = xrd_embeds + backbone.token_type_embeddings(
            torch.ones(xrd_embeds.shape[:2], device=device).long()
        )

        xrd_attention_masks = torch.ones(
            xrd_embeds.shape[:2],
            device=device
        )

        # -------------------------
        # concat
        # -------------------------
        x = torch.cat([mofid_embeds, xrd_embeds], dim=1)
        x_masks = torch.cat([attention_mask, xrd_attention_masks], dim=1)

        # -------------------------
        # transformer blocks
        # -------------------------
        for blk in backbone.vision_transformer.blocks:
            x, _ = blk(x, mask=x_masks)

        x = backbone.vision_transformer.norm(x)

        # -------------------------
        # pooling
        # -------------------------
        pooled = self.pool_with_mode(x, x_masks, mode=self.pooling_mode)
        return pooled

    # -------------------------
    # forward
    # -------------------------
    def forward(self, batch):
        pooled_feats = self.extract_pooled_feats(batch)
        pred = self.regression_head(pooled_feats).squeeze(-1)
        return pred

    # -------------------------
    # loss
    # -------------------------
    def compute_loss(self, pred, target):
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "huber":
            return F.huber_loss(pred, target, delta=self.huber_delta)
        elif self.loss_type == "smoothl1":
            return F.smooth_l1_loss(pred, target)
        elif self.loss_type == "mae":
            return F.l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    # -------------------------
    # common step
    # -------------------------
    def shared_step(self, batch, stage="train"):
        y = batch["regression"].float().to(self.device)
        y_norm = self.normalizer.encode(y)

        pred_norm = self(batch)
        loss = self.compute_loss(pred_norm, y_norm)

        pred = self.normalizer.decode(pred_norm).float()
        y_raw = y.float()

        mae = torch.mean(torch.abs(pred - y_raw))

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return {
            "loss": loss,
            "pred": pred.detach().cpu(),
            "target": y_raw.detach().cpu(),
        }

    # -------------------------
    # train / val / test
    # -------------------------
    def training_step(self, batch, batch_idx):
        out = self.shared_step(batch, stage="train")
        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self.shared_step(batch, stage="val")
        self.validation_step_outputs.append(out)

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        preds = torch.cat([x["pred"] for x in self.validation_step_outputs], dim=0).numpy()
        targets = torch.cat([x["target"] for x in self.validation_step_outputs], dim=0).numpy()

        val_r2 = r2_score(targets, preds)
        val_mae = mean_absolute_error(targets, preds)

        self.log("val/r2", val_r2, prog_bar=True, sync_dist=True)
        self.log("val/mae_sklearn", val_mae, prog_bar=False, sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        out = self.shared_step(batch, stage="test")
        self.test_step_outputs.append(out)

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            return

        preds = torch.cat([x["pred"] for x in self.test_step_outputs], dim=0).numpy()
        targets = torch.cat([x["target"] for x in self.test_step_outputs], dim=0).numpy()

        np.save(f"{self.name}_test_label.npy", np.array(targets))
        np.save(f"{self.name}_test_logit.npy", np.array(preds))

        test_r2 = r2_score(targets, preds)
        test_mae = mean_absolute_error(targets, preds)

        self.log("test/r2", test_r2, sync_dist=True)
        self.log("test/mae_sklearn", test_mae, sync_dist=True)

        print(f"[TEST] R2: {test_r2:.4f}, MAE: {test_mae:.4f}")

        self.test_step_outputs.clear()

    # -------------------------
    # optimizer
    # -------------------------
    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer