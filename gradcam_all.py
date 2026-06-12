"""
Grad-CAM visualizations for every gesture class.

Loads a trained CNN checkpoint and produces three kinds of visualization
covering all 10 gesture classes:

    gradcam_overview.png        2x5 grid: one example per class (slide-ready)
    gradcam_<class>.png         per-class: 6 examples per gesture (10 files)
    gradcam_misclassified.png   the model's worst mistakes, with Grad-CAM
                                showing where it was looking

"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CLASS_NAMES, get_dataloaders
from models import build_model

class GradCAM:
    """Standard Grad-CAM hooking a conv layer."""

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
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam, class_idx


def find_target_layer(model):
    """Return the last Conv2d in the model, or None if no conv layers exist."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv

def load_model_from_checkpoint(args, device):
    """Recreate the architecture from history.json so ablation weights load."""
    model_kwargs = {}
    history_path = Path(args.checkpoint).parent / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            cfg = json.load(f).get("config", {})
        if args.model == "custom":
            model_kwargs["use_bn"] = not cfg.get("no_bn", False)
            model_kwargs["dropout"] = cfg.get("dropout", 0.5)
        elif args.model == "mlp":
            model_kwargs["dropout"] = cfg.get("dropout", 0.5)
        if cfg.get("image_size"):
            args.image_size = cfg["image_size"]
        print(f"Architecture from config: kwargs={model_kwargs}")

    model = build_model(args.model, num_classes=10,
                        image_size=args.image_size, **model_kwargs).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    return model


def gather_predictions(model, loader, device):
    """Return (preds, labels, probs, images) for the whole loader. Images are
    kept as CPU tensors so we can re-display them later."""
    preds, labels, probs, images = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            logits = model(x_dev)
            p = F.softmax(logits, dim=1)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(y.numpy())
            probs.append(p.cpu().numpy())
            images.append(x.numpy())
    return (np.concatenate(preds),
            np.concatenate(labels),
            np.concatenate(probs),
            np.concatenate(images))

def undo_normalize(img_chw):
    """Undo the dataset's mean=0.5, std=0.5 normalization so the displayed
    grayscale image looks natural."""
    return np.clip(img_chw.squeeze() * 0.5 + 0.5, 0, 1)


def overlay(ax, img, cam, title=None, fontsize=9):
    ax.imshow(img, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.45)
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.axis("off")


def plot_overview(model, cam_tool, images, preds, labels, device, out_path,
                  examples_per_class=1):
    """2x5 grid: one correctly-classified example per class."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    axes = axes.flatten()

    for c in range(10):
        correct = np.where((labels == c) & (preds == c))[0]
        if len(correct) == 0:
            axes[c].text(0.5, 0.5, f"no correct\n{CLASS_NAMES[c]}",
                         ha="center", va="center")
            axes[c].axis("off")
            continue
        idx = int(correct[0])  # deterministic: first correct example for this class
        img = torch.tensor(images[idx]).unsqueeze(0).to(device)
        cam, _ = cam_tool(img, class_idx=c)
        overlay(axes[c], undo_normalize(images[idx]), cam,
                title=CLASS_NAMES[c], fontsize=11)

    fig.suptitle("Grad-CAM overview — one example per gesture class",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def plot_per_class(model, cam_tool, images, preds, labels, probs, device,
                   out_dir, examples_per_class=6):
    """One PNG per class showing N correct examples with Grad-CAM overlays."""
    for c in range(10):
        correct = np.where((labels == c) & (preds == c))[0]
        if len(correct) == 0:
            print(f"  ({CLASS_NAMES[c]}: no correct examples, skipping)")
            continue
        chosen = correct[:examples_per_class]
        n = len(chosen)
        fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
        if n == 1:
            axes = axes.reshape(2, 1)

        for j, idx in enumerate(chosen):
            img_t = torch.tensor(images[idx]).unsqueeze(0).to(device)
            cam, _ = cam_tool(img_t, class_idx=c)
            img = undo_normalize(images[idx])
            conf = probs[idx, c]

            axes[0, j].imshow(img, cmap="gray")
            axes[0, j].set_title(f"input  (p={conf:.2f})", fontsize=9)
            axes[0, j].axis("off")

            overlay(axes[1, j], img, cam, title="Grad-CAM", fontsize=9)

        fig.suptitle(f"Grad-CAM — class '{CLASS_NAMES[c]}'", fontsize=12)
        plt.tight_layout()
        out_path = out_dir / f"gradcam_{c:02d}_{CLASS_NAMES[c]}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {out_path}")


def plot_misclassified(model, cam_tool, images, preds, labels, probs, device,
                       out_path, top_k=8):
    """Top-k most-confident wrong predictions with Grad-CAM on the wrong class.
    This shows what the model latched onto when it was confidently wrong."""
    wrong = np.where(preds != labels)[0]
    if len(wrong) == 0:
        print("  (no misclassifications — model is perfect, skipping)")
        return
    confidences = probs[wrong, preds[wrong]]
    order = wrong[np.argsort(-confidences)]
    chosen = order[:min(top_k, len(order))]
    n = len(chosen)

    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, idx in enumerate(chosen):
        img_t = torch.tensor(images[idx]).unsqueeze(0).to(device)
        # Use the WRONG class — we want to see what convinced the model
        cam, _ = cam_tool(img_t, class_idx=int(preds[idx]))
        img = undo_normalize(images[idx])
        conf = probs[idx, preds[idx]]

        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(
            f"true: {CLASS_NAMES[labels[idx]]}\n"
            f"pred: {CLASS_NAMES[preds[idx]]} ({conf:.2f})",
            fontsize=9, color="crimson",
        )
        axes[0, j].axis("off")

        overlay(axes[1, j], img, cam,
                title=f"Grad-CAM on '{CLASS_NAMES[preds[idx]]}'", fontsize=9)

    fig.suptitle("Most confident misclassifications — what convinced the model?",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="custom",
                   choices=["linear", "logistic", "mlp", "small", "custom"])
    p.add_argument("--out-dir", default="gradcam_out")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--examples-per-class", type=int, default=6)
    p.add_argument("--top-k-mistakes", type=int, default=8)
    p.add_argument("--subject-split", action="store_true",
                   help="Use subject-level split (must match training)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    model = load_model_from_checkpoint(args, device)
    target = find_target_layer(model)
    if target is None:
        print(f"\nModel '{args.model}' has no Conv2d layers — Grad-CAM not "
              f"applicable. Only the conv models (custom, small) work here.")
        return
    cam_tool = GradCAM(model, target)


    _, _, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment=False,
        subject_split=args.subject_split,
        seed=args.seed,
    )

    print("Running inference on test set...")
    preds, labels, probs, images = gather_predictions(model, test_loader, device)
    acc = (preds == labels).mean()
    print(f"Test accuracy: {acc:.4f}  ({(preds == labels).sum()}/{len(labels)})")

    print("\nGenerating overview...")
    plot_overview(model, cam_tool, images, preds, labels, device,
                  out_dir / "gradcam_overview.png")

    print(f"\nGenerating per-class figures ({args.examples_per_class} examples each)...")
    plot_per_class(model, cam_tool, images, preds, labels, probs, device,
                   out_dir, examples_per_class=args.examples_per_class)

    print(f"\nGenerating misclassification figure (top {args.top_k_mistakes})...")
    plot_misclassified(model, cam_tool, images, preds, labels, probs, device,
                       out_dir / "gradcam_misclassified.png",
                       top_k=args.top_k_mistakes)

    print(f"\nAll Grad-CAM outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()