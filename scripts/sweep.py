"""
sweep.py — Short hyperparameter sweep comparing fusion strategies.

Sweeps four fusion methods for the Parallel CNN-GRU using the Validation set.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

import config
from models import build_model
from trainer import train_epoch, evaluate


def _run_single(
    cfg_overrides: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    sweep_epochs: int,
    lr: float,
) -> List[float]:
    """Train one sweep configuration and return per-epoch val accuracies."""
    merged_cfg = {**config.PARALLEL_CFG, **cfg_overrides}
    model = build_model(
        cfg=merged_cfg,
        window=config.WINDOW,
        n_classes=config.N_CLASSES,
        dropout=config.DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scaler    = GradScaler(enabled=(device.type == "cuda"))
    vl_accs: List[float] = []
    best_vl_acc, best_weights = 0.0, None

    for _ in tqdm(range(1, sweep_epochs + 1), desc=f"  epochs", leave=False, unit="ep"):
        train_epoch(model, train_loader, criterion, optimiser, scaler, device)
        _, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        vl_accs.append(vl_acc)
        
        if vl_acc > best_vl_acc:
            best_vl_acc = vl_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    del model, optimiser
    if device.type == "cuda": torch.cuda.empty_cache()
    elif device.type == "mps": torch.mps.empty_cache()

    return vl_accs, best_weights


def run_sweep(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device = config.DEVICE,
    sweep_epochs: int = config.SWEEP_EPOCHS,
    lr: float = config.LR,
    output_dir: str = config.OUTPUT_DIR,
) -> None:
    """Run all fusion sweep configurations and plot results."""
    results = []
    
    # Sweep configurations
    sweep_configs = [
        {"name": "concat",          "fusion": "concat"},
        {"name": "concat_fc",       "fusion": "concat_fc"},
        {"name": "concat_conv1d",   "fusion": "concat_conv1d"},
        {"name": "add (256)",       "fusion": "add", "cnn_fc": 256, "rnn_fc_in": 256, "rnn_fc_out": 256},
    ]

    cfg_bar = tqdm(sweep_configs, desc="Sweep configs", unit="cfg")
    for raw_cfg in cfg_bar:
        name = raw_cfg["name"]
        overrides = {k: v for k, v in raw_cfg.items() if k != "name"}
        cfg_bar.set_description(f"Sweep: {name}")
        
        vl_accs, best_weights = _run_single(overrides, train_loader, val_loader, device, sweep_epochs, lr)
        best = max(vl_accs)
        tqdm.write(f"  {name:25s}  best_val_acc={best:.4f}")
        
        result_item = {"name": name, "best_acc": best, "history": vl_accs}
        results.append(result_item)

        # ── Checkpoint: Save weights and intermediate results ──
        # Saves the best weights for THIS specific config
        fn_safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        ckpt_name = f"best_sweep_{fn_safe_name}.pt"
        torch.save(best_weights, os.path.join(output_dir, ckpt_name))
        
        # Save intermediate JSON for crash recovery
        import json
        with open(os.path.join(output_dir, "parallel_sweep_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        # ── Intermediate Plot ──
        _plot_sweep(results, sweep_epochs, output_dir)

    print("\nRanking (Based on Validation Accuracy):")
    for r in sorted(results, key=lambda x: x["best_acc"], reverse=True):
        print(f"  {r['name']:25s}  {r['best_acc']:.4f}")


def _plot_sweep(results: List[dict], sweep_epochs: int, output_dir: str) -> None:
    """Internal helper to refresh the sweep plot."""
    if not results: return
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        ax.plot(
            range(1, sweep_epochs + 1),
            [a * 100 for a in r["history"]],
            lw=2,
            label=f"{r['name']}  (best {r['best_acc'] * 100:.2f}%)",
        )
    ax.set(xlabel="Epoch", ylabel="Val Acc (%)", title="Parallel CNN-GRU — Fusion Sweep (Live Results)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "parallel_sweep.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
