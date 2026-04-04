"""
main.py — Entry point for EEG Motor Imagery Parallel CNN-GRU.

Updated for 3-way Subject-Independent Training:
    1. Subjects 1-10 -> Split into Train and Validation.
    2. Subjects 11-12 -> Reserved for final Blind Test.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
import config
from datasets    import build_loaders
from models      import build_model
from trainer     import run_training, evaluate
from evaluate_and_plot import run_final_evaluation
from sweep import run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG Motor Imagery — Parallel CNN-GRU (3-way split)")
    parser.add_argument("--epochs",  type=int,   default=config.EPOCHS, help="Epochs (default: %(default)d)")
    parser.add_argument("--batch",   type=int,   default=config.BATCH,  help="Batch size (default: %(default)d)")
    parser.add_argument("--lr",      type=float, default=config.LR,     help="Learning rate (default: %(default)s)")
    parser.add_argument("--subjects", type=int,  default=12,            help="Total subjects (default: 12)")
    parser.add_argument("--eval-only", action="store_true",             help="Skip training; load best and test")
    parser.add_argument("--sweep",     action="store_true",             help="Run fusion hyperparameter sweep")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = config.DEVICE

    print(f"\n" + "="*60 + "\n  SUBJECT-INDEPENDENT EXPERIMENT" + "\n" + "="*60)
    print(f"  Device           : {device}")
    print(f"  Total Subjects   : {args.subjects}")
    print(f"  - Train/Val Pool : Subjects 1-10")
    print(f"  - Blind Test Set : Subjects 11-12")
    print(f"  Max Epochs       : {args.epochs}")
    print(f"  Output Dir       : {config.OUTPUT_DIR}\n")

    # ── 1. Build 3-way DataLoaders ───────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = build_loaders(
        num_subjects=args.subjects,
        batch_size=args.batch,
    )

    # ── 2. Build Model ───────────────────────────────────────────────────────
    model = build_model(
        cfg=config.PARALLEL_CFG, window=config.WINDOW, 
        n_classes=config.N_CLASSES, dropout=config.DROPOUT
    ).to(device)

    # ── 3. Train on Train/Val Pool ───────────────────────────────────────────
    history = {"tr_loss": [], "vl_loss": [], "tr_acc": [], "vl_acc": []}
    best_val_acc = 0.0

    if args.sweep:
        print("\n" + "=" * 60)
        print("  FUSION HYPERPARAMETER SWEEP")
        print("=" * 60)
        run_sweep(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=config.OUTPUT_DIR,
        )
    elif not args.eval_only:
        try:
            history, best_val_acc = run_training(
                model=model, train_loader=train_loader, val_loader=val_loader,
                device=device, epochs=args.epochs, lr=args.lr, output_dir=config.OUTPUT_DIR
            )
        except Exception as e:
            # General exception still allows plotting if we have *something*
            print(f"\n⚠️ Unexpected error during training: {e}")

    else:
        print("[main] --eval-only: skipping training.")

    # ── 4. Final Blind Evaluation on Subjects 11-12 ──────────────────────────
    # We load the weights that did best on SUBJECTS 1-10 VALIDATION
    print(f"\n" + "-"*60 + "\n  FINAL BLIND TEST (Subjects 11-12)" + "\n" + "-"*60)
    
    # Using the existing plotter but we'll manually feed it the test set
    run_final_evaluation(
        model=model,
        test_loader=test_loader,  # Truly independent now (S11-S12)
        class_names=class_names,
        history=history if history["tr_loss"] else None, # plotter handles empty history
        best_acc=best_val_acc,
        device=device,
        output_dir=config.OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
