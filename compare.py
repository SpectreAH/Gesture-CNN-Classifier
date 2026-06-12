"""
Aggregate results from multiple training runs into one comparison table.

Scans a directory tree for history.json files (saved by Train.py), then prints
a sorted table and writes a CSV you can paste straight into your report.

Usage
-----
Compare everything under runs/:
    python compare.py --runs-dir runs

Only the custom-model runs (your ablations live here):
    python compare.py --runs-dir runs --model custom

Sort by validation accuracy instead of test accuracy:
    python compare.py --runs-dir runs --sort-by best_val_acc

Write the CSV somewhere specific:
    python compare.py --runs-dir runs --out report_table.csv
"""
import argparse
import csv
import json
from pathlib import Path


def find_runs(runs_dir):
    """Find every history.json under runs_dir and load its summary fields."""
    runs = []
    root = Path(runs_dir)
    for hp in sorted(root.rglob("history.json")):
        try:
            with open(hp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"  (skipping unreadable {hp})")
            continue
        cfg = data.get("config", {})
        runs.append({
            "name": str(hp.parent.relative_to(root)),
            "model": cfg.get("model", "?"),
            "bn": "no" if cfg.get("no_bn", False) else "yes",
            "aug": "no" if cfg.get("no_aug", False) else "yes",
            "split": "subj" if cfg.get("subject_split", False) else "rand",
            "lr": cfg.get("lr"),
            "dropout": cfg.get("dropout"),
            "weight_decay": cfg.get("weight_decay"),
            "optimizer": cfg.get("optimizer"),
            "batch_size": cfg.get("batch_size"),
            "epochs": cfg.get("epochs"),
            "num_params": data.get("num_params"),
            "best_val_acc": data.get("best_val_acc"),
            "test_acc": data.get("test_acc"),
            "test_loss": data.get("test_loss"),
            "train_time_min": data.get("train_time_min"),
        })
    return runs


def fmt(v, spec=""):
    if v is None:
        return "-"
    if spec:
        return format(v, spec)
    return str(v)


def print_table(runs, sort_key):
    runs = sorted(runs,
                  key=lambda r: (r.get(sort_key) is None, -(r.get(sort_key) or 0)))

    
    cols = [
        ("name",       24, lambda r: r["name"][:24]),
        ("model",      10, lambda r: r["model"]),
        ("split",       5, lambda r: r["split"]),
        ("BN",          4, lambda r: r["bn"]),
        ("aug",         4, lambda r: r["aug"]),
        ("lr",          9, lambda r: fmt(r["lr"], ".0e")),
        ("drop",        5, lambda r: fmt(r["dropout"])),
        ("opt",         5, lambda r: fmt(r["optimizer"])),
        ("params",     11, lambda r: f"{r['num_params']:,}" if r["num_params"] else "-"),
        ("val_acc",     8, lambda r: fmt(r["best_val_acc"], ".4f")),
        ("test_acc",    9, lambda r: fmt(r["test_acc"], ".4f")),
        ("time(m)",     8, lambda r: fmt(r["train_time_min"], ".1f")),
    ]

    header = "  ".join(f"{h:<{w}}" for h, w, _ in cols)
    print(header)
    print("-" * len(header))
    for r in runs:
        row = "  ".join(f"{fn(r):<{w}}" for _, w, fn in cols)
        print(row)
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs",
                   help="Directory to scan recursively for history.json files")
    p.add_argument("--model", default=None,
                   help="Only show runs for this model (e.g. custom)")
    p.add_argument("--split", default=None, choices=["rand", "subj"],
                   help="Only show runs from this split type")
    p.add_argument("--sort-by", default="test_acc",
                   choices=["test_acc", "best_val_acc",
                            "num_params", "train_time_min"])
    p.add_argument("--out", default=None,
                   help="CSV output path (default: <runs-dir>/comparison.csv)")
    args = p.parse_args()

    runs = find_runs(args.runs_dir)
    if args.model:
        runs = [r for r in runs if r["model"] == args.model]
    if args.split:
        runs = [r for r in runs if r["split"] == args.split]

    if not runs:
        filters = []
        if args.model: filters.append(f"model={args.model}")
        if args.split: filters.append(f"split={args.split}")
        msg = f"No runs found under {args.runs_dir}"
        if filters:
            msg += f" matching {', '.join(filters)}"
        print(msg)
        return

    print(f"\nFound {len(runs)} run(s) under {args.runs_dir}/\n")
    sorted_runs = print_table(runs, args.sort_by)

    out_path = Path(args.out) if args.out else Path(args.runs_dir) / "comparison.csv"
    fields = list(sorted_runs[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted_runs)
    print(f"\nCSV written to: {out_path}")

    best = sorted_runs[0]
    if best.get(args.sort_by) is not None:
        v = best[args.sort_by]
        v_str = f"{v:.4f}" if "acc" in args.sort_by else f"{v}"
        print(f"Top by {args.sort_by}: {best['name']} ({args.sort_by}={v_str})")


if __name__ == "__main__":
    main()