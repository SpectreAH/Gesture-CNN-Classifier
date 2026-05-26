"""

Produces (saved to --out-dir):
    confusion_matrix_norm.png      normalized confusion matrix
    confusion_matrix_raw.png       raw-count confusion matrix
    training_curves.png            train/val loss & accuracy over epochs
    per_class.json                 precision / recall / F1 per class
    top_misclassifications.json    most-confident wrong predictions
    gradcam.png                    Grad-CAM heatmaps on sample images(Only for CNNs)
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from dataset import CLASS_NAMES, get_dataloaders
from models import build_model


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        probs = F.softmax(out, dim=1)
        all_preds.append(out.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
        all_probs.append(probs.cpu().numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )



#Graphs the model
def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, out_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (row-normalized)" if normalize else ""))
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def per_class_report(y_true, y_pred, class_names, out_path):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    report = {
        cls: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i, cls in enumerate(class_names)
    }
    with open(out_path, "w") as fp:
        json.dump(report, fp, indent=2)
    return report


def find_misclassifications(y_true, y_pred, probs, top_k=20):
    """Return indices of the most confidently-wrong predictions for error analysis."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return []
    confidences = probs[wrong, y_pred[wrong]]  # confidence in the WRONG class
    sorted_idx = wrong[np.argsort(-confidences)]
    return sorted_idx[:top_k].tolist()


# Grad-CAM for CNN
class GradCAM:
    """Minimal Grad-CAM implementation for a chosen target conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        self.model.zero_grad()
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx


def _find_target_layer(model):
    """Return the last Conv2d in the model, or None for non-convolutional models."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv

#Will not work for non-convolutional models, but that's fine since Grad-CAM is only for CNNs
def plot_gradcam_examples(model, loader, device, out_path, n_examples=8):
    target = _find_target_layer(model)
    if target is None:
        print("Could not find a target conv layer for Grad-CAM; skipping.")
        return
    cam_tool = GradCAM(model, target)

    fig, axes = plt.subplots(2, n_examples, figsize=(2.5 * n_examples, 5))
    collected = 0
    for x, y in loader:
        for i in range(x.size(0)):
            if collected >= n_examples:
                break
            xi = x[i:i + 1].to(device)
            cam, pred = cam_tool(xi)
            img = x[i, 0].cpu().numpy()

            axes[0, collected].imshow(img, cmap="gray")
            axes[0, collected].set_title(
                f"true: {CLASS_NAMES[int(y[i])]}\npred: {CLASS_NAMES[pred]}",
                fontsize=9,
            )
            axes[0, collected].axis("off")

            axes[1, collected].imshow(img, cmap="gray")
            axes[1, collected].imshow(cam, cmap="jet", alpha=0.5)
            axes[1, collected].set_title("Grad-CAM", fontsize=9)
            axes[1, collected].axis("off")
            collected += 1
        if collected >= n_examples:
            break

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="custom",
                   choices=["linear", "logistic", "mlp", "small", "custom"])
    p.add_argument("--out-dir", default="runs/run/eval")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use same seed as training so the test split is identical
    _, _, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment=False,
        seed=args.seed,
    )

    model = build_model(args.model, num_classes=10,
                        image_size=args.image_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("Running inference on test set...")
    preds, labels, probs = get_predictions(model, test_loader, device)

    # Overall and per-class metrics
    print("\n-------------Test Metrics ----------------")
    print(classification_report(labels, preds, target_names=CLASS_NAMES,
                                digits=4, zero_division=0))
    per_class_report(labels, preds, CLASS_NAMES, out_dir / "per_class.json")

    # Confusion matrices
    plot_confusion_matrix(labels, preds, CLASS_NAMES,
                          out_dir / "confusion_matrix_norm.png", normalize=True)
    plot_confusion_matrix(labels, preds, CLASS_NAMES,
                          out_dir / "confusion_matrix_raw.png", normalize=False)

    # Training curves
    history_path = Path(args.checkpoint).parent / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)["history"]
        plot_training_curves(history, out_dir / "training_curves.png")

    #most-confident wrong predictions
    wrong_idx = find_misclassifications(labels, preds, probs, top_k=20)
    with open(out_dir / "top_misclassifications.json", "w") as f:
        json.dump({
            "indices": wrong_idx,
            "true_labels": [CLASS_NAMES[labels[i]] for i in wrong_idx],
            "predicted_labels": [CLASS_NAMES[preds[i]] for i in wrong_idx],
            "confidences": [float(probs[i, preds[i]]) for i in wrong_idx],
        }, f, indent=2)

    # Grad-CAM
    plot_gradcam_examples(model, test_loader, device,
                          out_dir / "gradcam.png", n_examples=8)

    print(f"\nAll evaluation outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()