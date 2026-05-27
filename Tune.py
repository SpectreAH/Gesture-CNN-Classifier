"""
These should be the main tunes as to determine the hyperparameters. Should be able to find the best ones.

# Main effort: tune the custom CNN (~2 hrs for 27 trials, or use random search to cut it)
python tune.py --data-root archive/leapGestRecog --model custom \
    --out-dir runs/tune_custom --epochs 10 \
    --lr 1e-2 1e-3 1e-4 --dropout 0.0 0.3 0.5 --weight-decay 1e-3 1e-4 1e-5

# Secondary: tune the MLP (~30 min for 6 trials)
python tune.py --data-root archive/leapGestRecog --model mlp \
    --out-dir runs/tune_mlp --epochs 10 \
    --lr 1e-2 1e-3 1e-4 --dropout 0.2 0.5

# Quick lr sweeps for the shallow baselines (~15 min each)
python tune.py --data-root archive/leapGestRecog --model logistic \
    --out-dir runs/tune_logistic --epochs 10 --lr 1e-1 1e-2 1e-3 1e-4

python tune.py --data-root archive/leapGestRecog --model linear \
    --out-dir runs/tune_linear --epochs 10 --lr 1e-1 1e-2 1e-3 1e-4
"""
import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    # --- required ---
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", required=True,
                   choices=["linear", "logistic", "mlp", "small", "custom"])

    # --- run management ---
    p.add_argument("--out-dir", default="runs/tune",
                   help="Parent directory for all trial subfolders")
    p.add_argument("--train-script", default="Train.py",
                   help="Path to the training script (default: Train.py)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Epochs PER TRIAL — keep small while tuning")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)

    # --- search strategy ---
    p.add_argument("--mode", choices=["grid", "random"], default="grid",
                   help="grid = try every combination; random = sample n-trials")
    p.add_argument("--n-trials", type=int, default=None,
                   help="(random mode) number of trials to run")
    p.add_argument("--metric", default="val_acc",
                   choices=["val_acc", "test_acc"],
                   help="Metric to sort the summary by")

    # --- hyperparameter search spaces ---
    p.add_argument("--lr", type=float, nargs="+", default=[1e-3],
                   help="Learning rates to try (one or more values)")
    p.add_argument("--batch-size", type=int, nargs="+", default=[64],
                   help="Batch sizes to try")
    p.add_argument("--dropout", type=float, nargs="+", default=[0.5],
                   help="Dropout values to try")
    p.add_argument("--weight-decay", type=float, nargs="+", default=[1e-4],
                   help="Weight-decay values to try")
    p.add_argument("--optimizer", nargs="+", default=["adam"],
                   help="Optimizers to try (adam, sgd)")

    # --- flags that apply to every trial in the sweep ---
    p.add_argument("--no-bn", action="store_true",
                   help="(custom only) disable BN for all trials")
    p.add_argument("--no-aug", action="store_true",
                   help="disable data augmentation for all trials")

    # --- misc ---
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip trials whose history.json already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without running")
    return p.parse_args()


def make_configs(args):
    """Yield (trial_name, hyperparam_dict) for each trial."""
    space = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
    }
    keys = list(space.keys())
    all_combos = list(itertools.product(*(space[k] for k in keys)))

    if args.mode == "random":
        n = args.n_trials or min(10, len(all_combos))
        random.seed(args.seed)
        combos = random.sample(all_combos, min(n, len(all_combos)))
    else:
        combos = all_combos

    # Only include a hyperparam in the trial name if it's actually being varied,
    # so single-knob sweeps have short, readable names.
    varied = {k for k, v in space.items() if len(v) > 1}

    for combo in combos:
        cfg = dict(zip(keys, combo))
        parts = []
        if "lr" in varied:           parts.append(f"lr{cfg['lr']:.0e}")
        if "batch_size" in varied:   parts.append(f"bs{cfg['batch_size']}")
        if "dropout" in varied:      parts.append(f"do{cfg['dropout']}")
        if "weight_decay" in varied: parts.append(f"wd{cfg['weight_decay']:.0e}")
        if "optimizer" in varied:    parts.append(f"opt{cfg['optimizer']}")
        name = "_".join(parts) if parts else "default"
        yield name, cfg


def build_command(args, cfg, out_dir):
    """Build the Train.py command for one trial."""
    cmd = [
        sys.executable, args.train_script,
        "--data-root", args.data_root,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--image-size", str(args.image_size),
        "--seed", str(args.seed),
        "--num-workers", str(args.num_workers),
        "--out-dir", str(out_dir),
        "--lr", str(cfg["lr"]),
        "--batch-size", str(cfg["batch_size"]),
        "--dropout", str(cfg["dropout"]),
        "--weight-decay", str(cfg["weight_decay"]),
        "--optimizer", cfg["optimizer"],
    ]
    if args.no_bn:
        cmd.append("--no-bn")
    if args.no_aug:
        cmd.append("--no-aug")
    return cmd


def main():
    args = parse_args()

    if not Path(args.train_script).exists():
        print(f"ERROR: train script not found at '{args.train_script}'.")
        print(f"Hint: pass --train-script with the right path "
              f"(e.g. 'train.py' or 'Train.py').")
        sys.exit(1)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    configs = list(make_configs(args))
    print(f"\n{'='*70}")
    print(f"Hyperparameter sweep — {args.mode} search")
    print(f"Model: {args.model}  |  Epochs/trial: {args.epochs}  |  "
          f"Trials: {len(configs)}")
    print(f"Output: {out_root}")
    print(f"{'='*70}")
    for name, cfg in configs:
        print(f"  {name:<35s}  {cfg}")

    if args.dry_run:
        print("\n(dry run — exiting without running anything)")
        return

    # ---- run all trials sequentially ----
    results = []
    for i, (name, cfg) in enumerate(configs, 1):
        trial_dir = out_root / name
        history_path = trial_dir / "history.json"

        print(f"\n{'='*70}")
        print(f"Trial {i}/{len(configs)}: {name}")
        print(f"{'='*70}")

        if args.skip_existing and history_path.exists():
            print(f"  -> already done, skipping (delete {history_path} to re-run)")
        else:
            cmd = build_command(args, cfg, trial_dir)
            print(f"  $ {' '.join(cmd)}\n")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  !! trial failed with exit code {e.returncode}, continuing")
                continue
            except KeyboardInterrupt:
                print("\n\nInterrupted by user — summarising what we have so far...")
                break

        # ---- collect result ----
        if history_path.exists():
            with open(history_path) as f:
                h = json.load(f)
            results.append({
                "trial": name,
                **cfg,
                "best_val_acc": h.get("best_val_acc"),
                "test_acc": h.get("test_acc"),
                "test_loss": h.get("test_loss"),
                "train_time_min": h.get("train_time_min"),
                "num_params": h.get("num_params"),
            })

    # ---- summary ----
    if not results:
        print("\nNo successful results to summarise.")
        return

    metric_key = "best_val_acc" if args.metric == "val_acc" else "test_acc"
    results.sort(key=lambda r: r.get(metric_key) or 0, reverse=True)

    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*70}")
    print(f"SUMMARY  (sorted by {metric_key}, best first)")
    print(f"{'='*70}")
    print(f"{'trial':<35} {'val_acc':>9} {'test_acc':>9} {'time(min)':>10}")
    print("-" * 70)
    for r in results:
        va = f"{r['best_val_acc']:.4f}" if r.get('best_val_acc') is not None else "  -  "
        ta = f"{r['test_acc']:.4f}"     if r.get('test_acc')     is not None else "  -  "
        tm = f"{r['train_time_min']:.1f}" if r.get('train_time_min') is not None else "  -  "
        print(f"{r['trial']:<35} {va:>9} {ta:>9} {tm:>10}")

    best = results[0]
    print(f"\nBest trial: {best['trial']}")
    print(f"  {metric_key} = {best[metric_key]:.4f}")
    print(f"  lr={best['lr']}, batch_size={best['batch_size']}, "
          f"dropout={best['dropout']}, weight_decay={best['weight_decay']}, "
          f"optimizer={best['optimizer']}")
    print(f"\nFull summary: {csv_path}")
    print(f"To get plots/confusion-matrix for the best trial:")
    print(f"  python Evaluate.py --data-root {args.data_root} \\")
    print(f"      --checkpoint {out_root}/{best['trial']}/best.pt \\")
    print(f"      --model {args.model} --out-dir {out_root}/{best['trial']}/eval")


if __name__ == "__main__":
    main()