# EXIT: Experimental XRD Integrated Transformer

A multimodal transformer for MOF property prediction that fuses **experimental XRD patterns** and **MOFid strings** to predict surface area (SA) and pore volume (PV).

> Kang et al. *J. Am. Chem. Soc.* 147, 5, 3943–3958 (2025)

---

## Why EXIT?

Existing ML models for MOF property prediction use structure-derived descriptors (MOFid, graph features) that are identical for the same MOF regardless of experimenter — so they produce identical predictions even when the same MOF synthesized under different conditions has different properties.

EXIT addresses this by incorporating the **experimental XRD pattern** as an additional input:

| Input | Type | Description |
|---|---|---|
| **MOFid** | Simulated | Encodes metal node, organic linker, topology — same for all samples of a given MOF |
| **Experimental XRD** | Experimental | Reflects actual crystal structure and synthesis conditions — varies between samples of the same MOF |

---

## Architecture

```
xrd  [B, 1, 4500]  ──► VisionTransformer1D (patch_size=20) ──► [B, 226, 768]  ─┐
                                                                                   ├─► concat + token-type embed
mofid [B, seq_len] ──► MOFidEncoder (embed + pos_enc)       ──► [B, seq_len, 768] ┘
                                                                                   │
                                                              Shared Transformer Blocks (6 layers, 8 heads)
                                                                                   │
                                                              CLS token ──► RegressionHead ──► SA or PV
```

- **VisionTransformer1D**: patches the 1D XRD signal (seq_length=4500, patch_size=20 → 225 patches + 1 CLS token)
- **MOFidEncoder**: token embedding + sinusoidal positional encoding; self-attention is handled jointly in shared blocks
- **Shared Transformer Blocks**: process the concatenated MOFid+XRD sequence (cross-modal attention)
- **Task Heads**: dynamically activated via `loss_names` in config (regression, mofid MLM, xrd reconstruction, vf)

---

## Installation

```bash
git clone https://github.com/seunghhs/EXIT.git
cd EXIT
pip install -e .
```

---

## Data Format

Each dataset is a `.pkl` file containing a list of dicts. Required keys:

**Pretraining (hMOF, simulated XRD):**
```python
{
    'xrd':   np.ndarray,  # shape [4500], normalized to [0, 1]
    'vf':    float,       # void fraction
    'mofid': str,         # e.g. "[Zn][Zn]...&&pcu"
    'name':  str,
    'ref':   str,
}
```

**Finetuning (experimental MOFs):**
```python
{
    'xrd':        np.ndarray,  # shape [4500], normalized to [0, 1]
    'regression': float,       # SA [m²/g] or PV [cm³/g]
    'mofid':      str,
    'name':       str,
    'ref':        str,
}
```

Normalization stats used internally:
- SA: mean=1288.56 m²/g, std=706.28 m²/g
- PV: mean=0.6558 cm³/g, std=0.3715 cm³/g

---

## Pretraining

Pretrain on hypothetical MOFs (hMOF database) with simulated XRD.

```bash
cd EXIT/exit

python pretrain.py \
    --config config/pretrain.yml \
    --devices 1 \
    --epoch 100 \
    --log_dir ./logs_pretrain \
    --ckpt_dir ./ckpt_pretrain
```

**Key config options (`config/pretrain.yml`):**
```yaml
dataset:
    train_data_dir: /path/to/train.pkl
    test_data_dir:  /path/to/test.pkl

model:
    seq_length: 4500    # XRD signal length
    patch_size: 20      # XRD patch size (4500/20 = 225 patches)
    embed_dim: 768
    hidden_dim: 768
    ntoken: 4021        # MOFid vocabulary size
    nhead: 8
    nlayers: 6

loss_names:
    mofid: 1            # MOFid masked language modeling (MLM, 15% masking)
    vf: 0               # void fraction regression
    xrd: 0              # XRD patch reconstruction
    regression: 0
    classification: 0

resume_from:            # path to checkpoint to resume from (null = train from scratch)
learning_rate: 0.0001
batch_size: 192
per_gpu_batchsize: 16
optim_type: adamw
warmup_steps: 0.05      # fraction of total steps
decay_power: 1          # polynomial decay
```

**What happens during pretraining:**
- 15% of MOFid tokens are randomly masked (`DataCollatorForLanguageModeling`)
- Model learns to reconstruct masked tokens (MLM objective)
- Checkpoint saved when `val/the_metric` (weighted sum of task losses) decreases
- Early stopping patience: 10 epochs

**Pretrained checkpoint:** `ckpt/epoch=99-step=368700.ckpt`

---

## Finetuning

Finetune on experimental MOF data for SA or PV prediction. Requires a pretrained checkpoint.

```bash
cd EXIT/exit

python finetune.py \
    --config config/finetune.yml \
    --devices 1 \
    --epoch 20 \
    --log_dir ./logs_finetune \
    --ckpt_dir ./ckpt_finetune
```

**Finetune config template (`config/finetune.yml`):**
```yaml
dataset:
    train_data_dir: /path/to/sa_train_3.pkl
    valid_data_dir: /path/to/sa_valid_3.pkl
    test_data_dir:  /path/to/sa_test_3.pkl

model:
    seq_length: 4500
    patch_size: 20
    embed_dim: 768
    hidden_dim: 768
    ntoken: 4021
    nhead: 8
    nlayers: 6

loss_names:
    regression: 1       # single-task regression
    mofid: 0
    vf: 0
    xrd: 0
    classification: 0

resume_from: /path/to/epoch=99-step=368700.ckpt   # pretrained checkpoint

regression_mean: 1288.56    # SA mean [m²/g]  |  PV: 0.6558 [cm³/g]
regression_std: 706.28      # SA std  [m²/g]  |  PV: 0.3715 [cm³/g]

learning_rate: 0.0001
batch_size: 128
per_gpu_batchsize: 128
optim_type: adamw
warmup_steps: 0.05
decay_power: 1

name: ft3               # prefix for saved .npy output files
seed: 0
visualize: False
num_nodes: 1
lr_mult: 1
end_lr: 0
```

**What happens during finetuning:**
- Pretrained weights loaded via `resume_from` (`strict=False`, so head weight mismatch is ignored)
- No MLM masking applied (`mlm=False`)
- Checkpoint saved when `val/the_metric_2` (negated MAE, higher is better) increases
- Early stopping patience: 5 epochs
- After training, test evaluation saves `{name}_test_label.npy` and `{name}_test_logit.npy`

**Run test evaluation on best checkpoint:**
```bash
python finetune.py \
    --config config/finetune.yml \
    --is_test True \
    --ckpt_dir ./ckpt_finetune
```

---

## Results

### Best Single Split

| Experiment | R² | MAE | Pearson |
|---|---|---|---|
| SA + Experimental XRD | **0.531** | **333.7 m²/g** | **0.770** |
| SA + Simulated XRD | 0.303 | 405.1 m²/g | 0.571 |
| PV + Experimental XRD | **0.585** | **0.217 cm³/g** | **0.791** |
| PV + Simulated XRD | 0.121 | 0.259 cm³/g | 0.601 |

### 5-Fold CV Average

| Experiment | R² | MAE | Pearson |
|---|---|---|---|
| SA + Experimental XRD | 0.36 ± 0.20 | 386.95 ± 60.20 m²/g | 0.65 ± 0.13 |
| SA + Simulated XRD | 0.25 ± 0.07 | 416.80 ± 12.41 m²/g | 0.57 ± 0.03 |
| PV + Experimental XRD | 0.38 ± 0.13 | 0.25 ± 0.02 cm³/g | 0.73 ± 0.05 |
| PV + Simulated XRD | 0.13 ± 0.08 | 0.27 ± 0.02 cm³/g | 0.61 ± 0.08 |

Experimental XRD consistently outperforms simulated XRD. The performance gap is largest for PV (R² 0.585 → 0.121), confirming that experimental XRD captures synthesis-condition-dependent structural variation that simulated XRD cannot.

---

## Project Structure

```
EXIT/
├── setup.py
├── exit/
│   ├── pretrain.py                    # pretraining entry point
│   ├── finetune.py                    # finetuning entry point
│   ├── config/
│   │   ├── pretrain.yml               # default pretrain config (mofid MLM task)
│   │   └── pretrain_sa.yml            # SA-focused pretrain config
│   ├── modules/
│   │   ├── model.py                   # MultiModal — core PyTorch Lightning module
│   │   ├── visiontransformer.py       # VisionTransformer1D — XRD patch encoder
│   │   ├── mofidtransformer.py        # MOFidEncoder — token embedding + positional encoding
│   │   ├── heads.py                   # task-specific prediction heads
│   │   ├── utils.py                   # loss functions, metrics, scheduler, normalizer
│   │   ├── metrics.py                 # custom TorchMetrics (Accuracy, Scalar)
│   │   └── additive.py                # alternative finetuning module (MultiModalAdditiveRegressor)
│   ├── dataset/
│   │   ├── dataset.py                 # pretraining dataset (xrd + vf + mofid)
│   │   ├── dataset_finetune.py        # finetuning dataset for regression
│   │   └── dataset_finetune_class.py  # finetuning dataset for classification
│   └── tokenizer/
│       ├── mof_tokenizer.py           # MOFTokenizer (SMILES regex + longest-match metadata)
│       └── vocab_new.txt              # vocabulary (~4021 tokens)
```
