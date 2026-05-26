# Hand Gesture Classification with Deep CNNs

Project for Group 2 final project. Builds a deep CNN to
classify 10 hand gestures from the **LeapGestRecog** near-infrared dataset
and is set up to support the full analysis required by the :
model comparison, error analysis, and ablation studies.

## Files

| File                  | Purpose                                                  |
| --------------------- | -------------------------------------------------------- |
| `dataset.py`          | LeapGestRecog `Dataset` + train/val/test DataLoaders     |
| `models.py`           | Custom CNN + SmallCNN + Linear + Logistic + MLP         |
| `train.py`            | Training script    |
| `evaluate.py`         | Confusion matrix, per-class metrics, Grad-CAM (only for CNN), errors    |
| `requirements.txt`    | Python dependencies                                      |

## Setup

```bash
#Threre is a virtual environement so you should be just able to activate it and then use it.
# If it is not working , a requirements textfile is available to use.

# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Get the dataset (Kaggle account required)
#    https://www.kaggle.com/datasets/gti-upm/leapgestrecog
#    Unzip so you have a folder structure like:
#        leapGestRecog/00/01_palm/*.png
#        leapGestRecog/00/02_l/*.png
#        ...
#        leapGestRecog/09/10_down/*.png
```

## Quick start

Train the main custom CNN, then evaluate it:


```bash


DATA=/path/to/leapGestRecog

# Train (~10-20 min on a single GPU for 20 epochs)
python train.py --data-root $DATA --out-dir runs/main --model custom

# Evaluate (produces all plots needed for the report)
python evaluate.py --data-root $DATA --checkpoint runs/main/best.pt \
                   --model custom --out-dir runs/main/eval
```

The eval step produces, in `runs/main/eval/`:
- `confusion_matrix_norm.png`, `confusion_matrix_raw.png`
- `training_curves.png`
- `per_class.json` — precision/recall/F1 per gesture
- `top_misclassifications.json` — most-confident wrong predictions
- `gradcam.png` — Grad-CAM heatmaps on sample test images (Only for CNN)

## Full experiment suite

```bash
# Model comparison (use SAME settings, only --model changes)


# Ablations on the custom model
python train.py --data-root $DATA --out-dir runs/no_bn       --no-bn
python train.py --data-root $DATA --out-dir runs/no_aug      --no-aug
python train.py --data-root $DATA --out-dir runs/no_dropout  --dropout 0
python train.py --data-root $DATA --out-dir runs/sgd         --optimizer sgd --lr 0.01
```

Each run produces `best.pt` (best-val-acc checkpoint) and `history.json`
(per-epoch metrics + config + final test accuracy). Run `evaluate.py` on
any of them to get its plots.

## Notes / known issues

- **Random split caveat.** The dataset has 10 subjects and the default
  splitter mixes them across train/val/test. Same-subject leakage inflates
  accuracy. (Need to adrress it later)

- **Reproducibility.** All scripts seed `torch.manual_seed(42)` by default;
  the same seed is used in `evaluate.py` so the test split should match the training.