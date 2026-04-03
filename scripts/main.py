"""
main.py — Entry point for EEG Motor Imagery Parallel CNN-RNN.

Usage:
    python scripts/main.py [--sweep] [--eval-only] [options]

All hyperparameters are read from config.py.

Typical workflow on M2 Mac:
    # Train
    python scripts/main.py

    # Train + fusion sweep
    python scripts/main.py --sweep

    # Evaluate only (load best checkpoint, generate plots)
    python scripts/main.py --eval-only
"""

import argparse
import os
import sys

# Ensure the scripts/ folder is on the path whether run from the repo root
# or from inside scripts/ itself.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import config
from datasets import build_loaders
from models import build_model
from trainer import run_training
from evaluate_and_plot import run_final_evaluation
from sweep import run_sweep


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEG Motor Imagery — Parallel CNN-RNN (Mac M2 edition)"
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help="Number of training epochs (default: %(default)d)",
    )
    parser.add_argument(
        "--batch", type=int, default=config.BATCH,
        help="Mini-batch size (default: %(default)d)",
    )
    parser.add_argument(
        "--lr", type=float, default=config.LR,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--subjects", type=int, default=config.NUM_SUBJECTS,
        help="Number of subjects (1-108) to include (default: %(default)d)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run fusion hyperparameter sweep after training",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load best checkpoint and generate evaluation plots",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override the data directory from config.py",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def print_system_info(device: torch.device) -> None:
    print("=" * 60)
    print("  EEG Motor Imagery — Parallel CNN-RNN")
    print("=" * 60)
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        print(f"  VRAM    : {total:.0f} MB  ({total / 1024:.1f} GB)")
    elif device.type == "mps":
        print("  Backend : Apple MPS (Metal Performance Shaders)")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args    = parse_args()
    device  = config.DEVICE

    data_dir = args.data_dir or config.PARALLEL_DATA_DIR

    print_system_info(device)
    print(f"\n  Data dir   : {data_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch}")
    print(f"  LR         : {args.lr}")
    print(f"  Subjects   : {args.subjects if args.subjects else 'All (108)'}")
    print(f"  Fusion     : {config.PARALLEL_CFG['fusion']}")
    print(f"  Output dir : {config.OUTPUT_DIR}\n")

    # ── Build DataLoaders ────────────────────────────────────────────────────
    train_loader, test_loader, class_names = build_loaders(
        data_dir=data_dir,
        num_subjects=args.subjects,
        batch_size=args.batch,
    )

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model(
        cfg=config.PARALLEL_CFG,
        window=config.WINDOW,
        n_classes=config.N_CLASSES,
        dropout=config.DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {n_params:,}  (~{n_params * 4 / 1024 ** 2:.1f} MB)\n")

    # ── Dry-run sanity check ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        cnn_x, rnn_x, _ = next(iter(train_loader))
        out = model(cnn_x.to(device, dtype=torch.float32), 
                    rnn_x.to(device, dtype=torch.float32))
    print(f"  Forward pass OK  : output shape {tuple(out.shape)}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    if not args.eval_only:
        history, best_acc = run_training(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            output_dir=config.OUTPUT_DIR,
            checkpoint_interval=config.CHECKPOINT_INTERVAL,
        )
    else:
        print("[main] --eval-only: skipping training, loading checkpoint.")
        history  = {"tr_loss": [], "te_loss": [], "tr_acc": [], "te_acc": []}
        best_acc = 0.0

    # ── Final evaluation & plots ─────────────────────────────────────────────
    run_final_evaluation(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        history=history,
        best_acc=best_acc,
        device=device,
        output_dir=config.OUTPUT_DIR,
    )

    # ── Optional fusion sweep ────────────────────────────────────────────────
    if args.sweep:
        print("\n" + "=" * 60)
        print("  FUSION HYPERPARAMETER SWEEP")
        print("=" * 60)
        run_sweep(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            output_dir=config.OUTPUT_DIR,
        )


if __name__ == "__main__":
    main()
