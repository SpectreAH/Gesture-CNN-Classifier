"""
Lines on command to run in the terminal 
--------
Main run (custom CNN with batch norm + augmentation):
    python train.py --data-root /path/to/leapGestRecog --out-dir runs/main

Ablations:
    python train.py --data-root /path/to/leapGestRecog --out-dir runs/no_bn   --no-bn
    python train.py --data-root /path/to/leapGestRecog --out-dir runs/no_aug  --no-aug
    python train.py --data-root /path/to/leapGestRecog --out-dir runs/sgd     --optimizer sgd --lr 0.01

Comparisons (shallow / non-conv baselines):
    python train.py --data-root /path/to/leapGestRecog --out-dir runs/linear    --model linear
    python train.py --data-root /path/to/leapGestRecog --out-dir runs/logistic  --model logistic
    python train.py --data-root /path/to/leapGestRecog --out-dir runs/mlp       --model mlp
    python train.py --data-root /path/to/leapGestRecog --out-dir runs/small     --model small
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from models import REGRESSION_MODELS, build_model, count_params


class MSEOneHotLoss(nn.Module):
    """MSE loss against one-hot targets used for linear regression baseline."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, target):
        one_hot = F.one_hot(target, self.num_classes).float()
        return F.mse_loss(logits, one_hot)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Path to leapGestRecog/")
    p.add_argument("--model", default="custom",
                   choices=["linear", "logistic", "mlp", "small", "custom"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-bn", action="store_true",
                   help="(ablation) disable BatchNorm in the custom CNN")
    p.add_argument("--no-aug", action="store_true",
                   help="(ablation) disable data augmentation")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--out-dir", default="runs/run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subject-split", action="store_true",
                   help="Subject-level split (no person appears in two splits)")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augment=not args.no_aug,
        seed=args.seed,
        subject_split=args.subject_split, 
    )
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    model_kwargs = {}
    if args.model == "custom":
        model_kwargs = dict(use_bn=not args.no_bn, dropout=args.dropout)
    elif args.model == "mlp":
        model_kwargs = dict(dropout=args.dropout)
    model = build_model(args.model, num_classes=10,
                        image_size=args.image_size, **model_kwargs).to(device)
    print(f"Model: {args.model} | Trainable params: {count_params(model):,}")

    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr,
                         weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9,
                        weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.model in REGRESSION_MODELS:
        criterion = MSEOneHotLoss(num_classes=10)
        print(f"Loss: MSE on one-hot targets")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"Loss: CrossEntropy")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate_loss_acc(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{time.time() - t0:.1f}s")

    total_time = time.time() - start
    print(f"\nTraining complete in {total_time / 60:.1f} min. "
          f"Best val acc: {best_val_acc:.4f}")

    model.load_state_dict(torch.load(out_dir / "best.pt"))
    te_loss, te_acc = evaluate_loss_acc(model, test_loader, criterion, device)
    print(f"Test loss {te_loss:.4f} | Test acc {te_acc:.4f}")

    with open(out_dir / "history.json", "w") as f:
        json.dump({
            "config": vars(args),
            "history": history,
            "best_val_acc": best_val_acc,
            "test_loss": te_loss,
            "test_acc": te_acc,
            "train_time_min": total_time / 60,
            "num_params": count_params(model),
        }, f, indent=2)
    print(f"Saved results to {out_dir}/")


if __name__ == "__main__":
    main()