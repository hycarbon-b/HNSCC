# hnscc-nnunet

nnUNet v2 pipeline for 3D CT segmentation of head and neck squamous cell carcinoma (HNSCC) tumors.

## Setup

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

**CPU (debug / smoke test):**
```bash
uv venv --python 3.11
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

**GPU (CUDA 12.1, full training):**
```bash
uv venv --python 3.11
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

## Usage

```bash
# CPU smoke test (2D, 5 epochs)
.venv/bin/python train.py --images /path/to/images --masks /path/to/masks

# GPU full training (3d_fullres, 1000 epochs)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
    --images /path/to/images --masks /path/to/masks
```

`--images` and `--masks` must contain matching `*.nii.gz` filenames (sorted order).

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-id` | 100 | nnUNet dataset ID |
| `--dataset-name` | HNSCC_CT | Dataset name |
| `--fold` | 0 | Cross-validation fold |
| `--train-split` | 0.8 | Train/test split ratio |
| `--epochs` | auto | Override epoch count |
| `--config` | auto | Override nnUNet config (`2d` / `3d_fullres`) |
| `--force-preprocess` | — | Re-run plan_and_preprocess |

## Output

Results saved to `nnUNet_results/Dataset<ID>_<NAME>/nnUNetTrainer__nnUNetPlans__<config>/fold_<N>/`.
