"""
trainer.py — Epoch-level train/eval with Early Stopping and Validation monitoring.
"""

import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import config

History = Dict[str, List[float]]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n_samples = 0.0, 0, 0
    bar = tqdm(loader, desc="  train", leave=False, unit="batch")
    
    for cnn_x, rnn_x, labels in bar:
        cnn_x  = cnn_x.to(device, dtype=torch.float32, non_blocking=True)
        rnn_x  = rnn_x.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
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
        bar.set_postfix(loss=f"{total_loss/n_samples:.4f}", acc=f"{correct/n_samples:.4f}")

    return total_loss / n_samples, correct / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss, correct, n_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for cnn_x, rnn_x, labels in tqdm(loader, desc="   eval", leave=False, unit="batch"):
        cnn_x  = cnn_x.to(device, dtype=torch.float32, non_blocking=True)
        rnn_x  = rnn_x.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(cnn_x, rnn_x)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        n_samples  += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / n_samples, correct / n_samples, all_preds, all_labels


def _save_checkpoint(path: str, epoch: int, model: nn.Module, optimiser: torch.optim.Optimizer, 
                     scaler: GradScaler, best_val_acc: float, history: History) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimiser.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_acc":      best_val_acc,
        "history":           dict(history),
    }, path)


def _free_device_cache(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.empty_cache()
    elif device.type == "mps": torch.mps.empty_cache()


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,   # Now monitoring Val, not Test
    device: torch.device = config.DEVICE,
    epochs: int = config.EPOCHS,
    lr: float = config.LR,
    output_dir: str = config.OUTPUT_DIR,
    checkpoint_interval: int = config.CHECKPOINT_INTERVAL,
    early_stopping_patience: int = 15,
) -> Tuple[History, float]:
    criterion = nn.CrossEntropyLoss()
    # Adding Weight Decay (L2 regularisation) to combat overfitting
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) 
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    ckpt_path   = os.path.join(output_dir, "best_parallel.pt")
    resume_path = os.path.join(output_dir, "parallel_resume.pt")

    best_val_acc = 0.0
    history = defaultdict(list)
    start_epoch, patience_counter = 1, 0

    if os.path.exists(resume_path):
        print(f"[trainer] Resuming from {resume_path} …")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optim_state_dict"])
        if "scaler_state_dict" in ckpt: scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history      = defaultdict(list, ckpt["history"])

    model.to(device)
    print(f"\n[trainer] Training | epochs {start_epoch}→{epochs} | device={device}\n")

    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc="Epochs", unit="ep", initial=start_epoch-1, total=epochs)
    
    try:
        for epoch in epoch_bar:
            t0 = time.perf_counter()
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimiser, scaler, device)
            vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)

            history["tr_loss"].append(tr_loss); history["tr_acc" ].append(tr_acc)
            history["vl_loss"].append(vl_loss); history["vl_acc" ].append(vl_acc)

            new_best, ckpt_tag = "", ""
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save(model.state_dict(), ckpt_path)
                new_best, patience_counter = " 🌟", 0  # Reset patience
            else:
                patience_counter += 1

            if epoch % checkpoint_interval == 0 or epoch == epochs:
                snapshot = os.path.join(output_dir, f"parallel_epoch_{epoch}.pt")
                _save_checkpoint(snapshot, epoch, model, optimiser, scaler, best_val_acc, history)
                _save_checkpoint(resume_path, epoch, model, optimiser, scaler, best_val_acc, history)
                
                # ── Save CSV History ──
                csv_path = os.path.join(output_dir, "parallel_history.csv")
                import pandas as pd
                pd.DataFrame(dict(history)).to_csv(csv_path, index_label="epoch_idx")
                
                ckpt_tag = " 💾"

            _free_device_cache(device)
            epoch_bar.set_postfix(tr_acc=f"{tr_acc:.4f}", vl_acc=f"{vl_acc:.4f}", best=f"{best_val_acc:.4f}", s=f"{time.perf_counter()-t0:.1f}")
            
            if new_best or ckpt_tag:
                tqdm.write(f"  Ep {epoch:3d}/{epochs}{new_best}{ckpt_tag}")

            if patience_counter >= early_stopping_patience:
                tqdm.write(f"\n🛑 Early stopping triggered at epoch {epoch} (no improvement in {early_stopping_patience} eps)")
                break
    except KeyboardInterrupt:
        tqdm.write("\n⚠️ Training interrupted by user. Returning current history...")

    return dict(history), best_val_acc
