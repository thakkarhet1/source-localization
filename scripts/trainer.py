"""
trainer.py — Epoch-level train / evaluate functions and the main training loop.

M2 8 GB notes:
  • torch.mps.empty_cache() is called after each epoch on MPS to return
    metal heap memory to the system.
  • Checkpoint saving uses the standard PyTorch format; best weights and full
    resume snapshots are written every CHECKPOINT_INTERVAL epochs.
"""

import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import config

History = Dict[str, List[float]]


# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one full training epoch for the Parallel CNN-RNN model.

    Args:
        model     : ParallelCNNRNN
        loader    : training DataLoader (yields cnn_x, rnn_x, labels)
        criterion : loss function (CrossEntropyLoss)
        optimiser : Adam (or any other)
        device    : target device

    Returns:
        (mean_loss, accuracy) over the whole epoch
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    n_samples  = 0

    for cnn_x, rnn_x, labels in loader:
        cnn_x  = cnn_x.to(device, dtype=torch.float32, non_blocking=True)
        rnn_x  = rnn_x.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)   # faster & lower memory than zeroing

        amp_device = 'cpu' if device.type == 'mps' else device.type
        with autocast(device_type=amp_device):
            logits = model(cnn_x, rnn_x)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        n_samples  += labels.size(0)

    return total_loss / n_samples, correct / n_samples


# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate the model on an entire DataLoader (no gradients).

    Returns:
        (mean_loss, accuracy, predictions, true_labels)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    n_samples  = 0
    all_preds  : List[int] = []
    all_labels : List[int] = []

    for cnn_x, rnn_x, labels in loader:
        cnn_x  = cnn_x.to(device, dtype=torch.float32, non_blocking=True)
        rnn_x  = rnn_x.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(cnn_x, rnn_x)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        n_samples  += labels.size(0)
        all_preds  .extend(preds.cpu().tolist())
        all_labels .extend(labels.cpu().tolist())

    return total_loss / n_samples, correct / n_samples, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
def _save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    best_acc: float,
    history: History,
) -> None:
    """Write a full resume checkpoint to *path*."""
    torch.save(
        {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimiser.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_acc":         best_acc,
            "history":          dict(history),
        },
        path,
    )


def _free_device_cache(device: torch.device) -> None:
    """Release unused memory back to the OS / Metal runtime."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device = config.DEVICE,
    epochs: int = config.EPOCHS,
    lr: float = config.LR,
    output_dir: str = config.OUTPUT_DIR,
    checkpoint_interval: int = config.CHECKPOINT_INTERVAL,
) -> Tuple[History, float]:
    """Full training loop with resume support and periodic checkpointing.

    Checkpointing strategy:
        • best_parallel.pt      — best test-accuracy weights only (small)
        • parallel_resume.pt    — most recent full checkpoint for resuming
        • parallel_epoch_N.pt   — snapshot every `checkpoint_interval` epochs

    Args:
        model               : ParallelCNNRNN already moved to `device`
        train_loader        : training DataLoader
        test_loader         : evaluation DataLoader
        device              : torch.device
        epochs              : total number of epochs to train
        lr                  : learning rate
        output_dir          : directory where checkpoints will be saved
        checkpoint_interval : save a periodic snapshot every N epochs

    Returns:
        (history dict, best_acc)
    """
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    ckpt_path   = os.path.join(output_dir, "best_parallel.pt")
    resume_path = os.path.join(output_dir, "parallel_resume.pt")

    best_acc    = 0.0
    history: History = defaultdict(list)
    start_epoch = 1

    # ── Resume from checkpoint if available ──────────────────────────────────
    if os.path.exists(resume_path):
        print(f"[trainer] Resuming from {resume_path} …")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optim_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc    = ckpt["best_acc"]
        history     = defaultdict(list, ckpt["history"])
        print(f"[trainer] Resumed at epoch {start_epoch - 1} | best_acc={best_acc:.4f}")
    elif os.path.exists(ckpt_path):
        print(f"[trainer] No resume file found — loading best weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.to(device)
    print(f"\n[trainer] Training | epochs {start_epoch}→{epochs} | device={device}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimiser, scaler, device)
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, device)

        history["tr_loss"].append(tr_loss)
        history["tr_acc" ].append(tr_acc)
        history["te_loss"].append(te_loss)
        history["te_acc" ].append(te_acc)

        # ── Save best weights ────────────────────────────────────────────────
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  🌟 New best: {best_acc:.4f} → {ckpt_path}")

        # ── Periodic full checkpoint ─────────────────────────────────────────
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            snapshot_path = os.path.join(output_dir, f"parallel_epoch_{epoch}.pt")
            _save_checkpoint(snapshot_path, epoch, model, optimiser, scaler, best_acc, history)
            _save_checkpoint(resume_path,   epoch, model, optimiser, scaler, best_acc, history)
            print(f"  💾 Checkpoint: {snapshot_path}")

        # ── Release device memory each epoch (important for MPS) ─────────────
        _free_device_cache(device)

        # ── Progress log every epoch (ensures nothing is missed in short runs) ─
        elapsed = time.perf_counter() - t0
        print(
            f"  Ep {epoch:3d}/{epochs} | "
            f"tr={tr_loss:.4f}/{tr_acc:.4f} | "
            f"te={te_loss:.4f}/{te_acc:.4f} | "
            f"best={best_acc:.4f} | {elapsed:.1f}s"
        )

    print(f"\n✅  Training done. Best test accuracy: {best_acc:.4f} → {ckpt_path}")
    return dict(history), best_acc
