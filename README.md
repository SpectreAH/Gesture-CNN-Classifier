# Hand Gesture Classification with Deep CNNs

Project for Group 2 final project. Builds a deep CNN to
classify 10 hand gestures from the **LeapGestRecog** near-infrared dataset
and is set up to support the full analysis:
model comparison, error analysis, and ablation studies.

## Files

| File                  | Purpose                                                  |
| --------------------- | -------------------------------------------------------- |
| `dataset.py`          | LeapGestRecog `Dataset` + train/val/test DataLoaders     |
| `models.py`           | Custom CNN + SmallCNN + Linear + Logistic + MLP         |
| `Train.py`            | Training script    |
| `Evaluate.py`         | Confusion matrix, per-class metrics, Grad-CAM (only for CNN), errors    |
| `requirements.txt`    | Python dependencies                                      |
| `Tune.py`             | Hyperparameters search                      |


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
python Train.py --data-root $DATA --out-dir runs/main --model custom

# Evaluate (produces all plots needed for the report)
python Evaluate.py --data-root $DATA --checkpoint runs/main/best.pt \
                   --model custom --out-dir runs/main/eval
```

The eval step produces, in `runs/main/eval/`:
- `confusion_matrix_norm.png`, `confusion_matrix_raw.png`
- `training_curves.png`
- `per_class.json` — precision/recall/F1 per gesture
- `top_misclassifications.json` — most-confident wrong predictions
- `gradcam.png` — Grad-CAM heatmaps on sample test images (Only for CNN)

## Full experiment suite

Each run produces `best.pt` (best-val-acc checkpoint) and `history.json`
(per-epoch metrics + config + final test accuracy). Run `Evaluate.py` on
any of them to get its plots.

## Notes / known issues

- **Random split caveat.** The dataset has 10 subjects and the default
  splitter mixes them across train/val/test. Same-subject leakage inflates
  accuracy. (Addressed later with a subject-based split, see Key results.)

- **Reproducibility.** All scripts seed `torch.manual_seed(42)` by default;
  the same seed is used in `Evaluate.py` so the test split should match the training.

- **Data leakage.** A problem that arises is that the models test all the images doing so memorizing the "shape" of the hand which results in the 99.99 % accuracy and precision. But we know that it is not a real world application. So we should use a set for training one for testing and one for validating.
    
## Presentation

- [View the slides](slides/presentation.pdf)
- [Read the report](report/report.pdf)

## Key results

| Model | Random split | Subject split |
|---|---|---|
| Linear regression | 60% | 51% |
| Logistic regression | 77% | 63% |
| MLP | 90% | 73% |
| SmallCNN | 97% | 80% |
| **Custom CNN** | **100%** | **87%** |

- A random split puts the same people in training and test sets, which inflates accuracy by ~13 percentage points (subject leakage).
- After hyperparameter tuning (lr 1e-3, dropout 0.3, weight decay 1e-4, Adam), the custom CNN reached ~98% on the subject split.
- Ablations show BatchNorm matters most; dropout had no measurable effect.
- Grad-CAM confirms the model focuses on the hand, not the background.

## Dataset

This project uses the [LeapGestRecog dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) (~2 GB, not included in this repo).
1. Download it from Kaggle
2. Extract it so the folder structure is `archive/leapGestRecog/00/01_palm/...`
3. Run the training scripts as described above

## Team

Group project by Soen Souprayen, Alexandre Huet - Ajou University exchange semester.
