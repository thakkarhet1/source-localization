"""
train.py — Self-contained training script for EEG Motor Imagery (Parallel CNN-GRU).

Designed for a Google Cloud Compute VM with GPU.
Data is pulled from GCS on first run; subsequent runs skip the download.

Usage:
    python train.py [--epochs N] [--batch N] [--lr F] [--subjects N]
                    [--data_dir PATH] [--output_dir PATH]
                    [--gcs_bucket NAME] [--gcs_prefix PATH]
                    [--download] [--eval_only]
"""

import argparse
import datetime
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

# ── Paths / GCS config ────────────────────────────────────────────────────────
GCS_BUCKET      = "parallel_eeg_decoding"
GCS_DATA_PREFIX = "eeg_data/npz_4class_parallel_W10"
DATA_DIR        = "/home/jupyter/eeg_data/npz_4class_parallel_W10"
OUTPUT_DIR      = "/home/jupyter/eeg_results"

# ── Hyperparameters ───────────────────────────────────────────────────────────
WINDOW           = 10
N_CLASSES        = 4
EPOCHS           = 100
SEED             = 42
BATCH            = 32
LR               = 1e-4
DROPOUT          = 0.5
GRAD_ACCUM_STEPS = 4
LR_T_MAX         = EPOCHS
LR_ETA_MIN       = 1e-6
NOISE_STD        = 0.05
NUM_WORKERS      = 4
CHECKPOINT_INTERVAL = 10
EARLY_STOP_PATIENCE = 15

PARALLEL_CFG = dict(
    conv_channels=32,
    cnn_fc=512,
    n_electrodes=64,
    rnn_fc_in=512,
    gru_hidden=128,
    gru_layers=2,
    rnn_fc_out=512,
    fusion="concat_fc",
)


# ── GCS Download ──────────────────────────────────────────────────────────────
def download_gcs_prefix(bucket_name: str, prefix: str, local_dir: str) -> None:
    """Download all blobs under `prefix` in `bucket_name` to `local_dir`."""
    try:
        from google.cloud import storage
    except ImportError:
        print("ERROR: google-cloud-storage not installed. Run: pip install google-cloud-storage")
        sys.exit(1)

    client = storage.Client()
    blobs  = list(client.list_blobs(bucket_name, prefix=prefix))
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {len(blobs)} file(s) from gs://{bucket_name}/{prefix} → {local_dir}")
    for blob in tqdm(blobs, desc="Downloading"):
        rel_path   = os.path.relpath(blob.name, prefix)
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            blob.download_to_filename(local_path)

    print(f"Download complete. Files in {local_dir}")


# ── Model ─────────────────────────────────────────────────────────────────────
class ParallelCNNGRU(nn.Module):
    """Parallel CNN + GRU with configurable fusion.

    CNN branch : (B,W,1,H,Wd) → 2-D CNN per frame → sum over W → cnn_fc vector
    GRU branch : (B,W,n_electrodes) → linear projection → GRU → rnn_fc vector
    Fusion     : configurable combination of the two branch outputs
    """

    def __init__(
        self,
        window_size:   int   = 10,
        conv_channels: int   = 32,
        cnn_fc:        int   = 256,
        n_electrodes:  int   = 64,
        rnn_fc_in:     int   = 256,
        gru_hidden:    int   = 16,
        gru_layers:    int   = 2,
        rnn_fc_out:    int   = 256,
        n_classes:     int   = 4,
        dropout:       float = 0.5,
        fusion:        str   = "add",
        **kwargs,
    ):
        super().__init__()
        self.fusion = fusion
        ch = conv_channels

        self.cnn_features = nn.Sequential(
            nn.Conv2d(1,    ch,   3, padding=1), nn.BatchNorm2d(ch),   nn.ELU(inplace=True),
            nn.Conv2d(ch,   ch*2, 3, padding=1), nn.BatchNorm2d(ch*2), nn.ELU(inplace=True),
            nn.Conv2d(ch*2, ch*4, 3, padding=1), nn.BatchNorm2d(ch*4), nn.ELU(inplace=True),
            nn.Conv2d(ch*4, ch*8, 3, padding=1), nn.BatchNorm2d(ch*8), nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 8, cnn_fc),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

        self.bidirectional = kwargs.get("bidirectional", False)
        self.rnn_proj = nn.Sequential(
            nn.Linear(n_electrodes, rnn_fc_in),
            nn.ELU(inplace=True),
        )
        self.gru = nn.GRU(
            rnn_fc_in, gru_hidden, gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        rnn_in_size = gru_hidden * 2 if self.bidirectional else gru_hidden
        self.rnn_head = nn.Sequential(
            nn.Linear(rnn_in_size, rnn_fc_out),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

        if fusion == "concat":
            fused_dim = cnn_fc + rnn_fc_out
            self.fuse_layer = None
        elif fusion == "add":
            if cnn_fc != rnn_fc_out:
                raise ValueError(f"'add' fusion requires cnn_fc == rnn_fc_out, got {cnn_fc} vs {rnn_fc_out}")
            fused_dim = cnn_fc
            self.fuse_layer = None
        elif fusion == "concat_fc":
            fused_dim = cnn_fc + rnn_fc_out
            self.fuse_layer = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.ELU(inplace=True))
        elif fusion == "concat_conv1d":
            fused_dim = cnn_fc + rnn_fc_out
            self.fuse_layer = nn.Sequential(nn.Conv1d(fused_dim, fused_dim, kernel_size=1), nn.ELU(inplace=True))
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion!r}")

        self.readout = nn.Linear(fused_dim, n_classes)

    def forward(self, cnn_x: torch.Tensor, rnn_x: torch.Tensor) -> torch.Tensor:
        B, W, C, H, Wd = cnn_x.shape

        cnn_frames = self.cnn_fc(self.cnn_features(cnn_x.reshape(B * W, C, H, Wd)))
        cnn_out    = cnn_frames.reshape(B, W, -1).sum(dim=1)

        proj    = self.rnn_proj(rnn_x.reshape(B * W, -1)).reshape(B, W, -1)
        gru_out, _ = self.gru(proj)
        rnn_out = self.rnn_head(gru_out[:, -1, :])

        if self.fusion == "concat":
            fused = torch.cat([cnn_out, rnn_out], dim=1)
        elif self.fusion == "add":
            fused = cnn_out + rnn_out
        elif self.fusion == "concat_fc":
            fused = self.fuse_layer(torch.cat([cnn_out, rnn_out], dim=1))
        elif self.fusion == "concat_conv1d":
            cat   = torch.cat([cnn_out, rnn_out], dim=1).unsqueeze(2)
            fused = self.fuse_layer(cat).squeeze(2)

        return self.readout(fused)


def build_model(cfg: dict, window: int, n_classes: int, dropout: float) -> ParallelCNNGRU:
    return ParallelCNNGRU(window_size=window, n_classes=n_classes, dropout=dropout, **cfg)


# ── Dataset ───────────────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, data_dir: str, num_subjects: int = 12):
        labels_path = os.path.join(data_dir, "S001_S108_win10_labels.npz")
        cnn_path    = os.path.join(data_dir, "S001_S108_win10_cnn_data.npz")
        rnn_path    = os.path.join(data_dir, "S001_S108_win10_rnn_data.npz")

        missing = [p for p in [labels_path, cnn_path, rnn_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing .npz files in {data_dir}: {missing}")

        print("[EEGDataset] Loading labels...")
        labels_raw   = np.load(labels_path, allow_pickle=True)["labels"]
        self.classes = sorted(set(labels_raw))
        c2i          = {c: i for i, c in enumerate(self.classes)}
        self.labels  = np.array([c2i[l] for l in labels_raw], dtype=np.int64)

        print("[EEGDataset] Loading CNN data...")
        self.cnn_data = np.load(cnn_path)["data"]

        print("[EEGDataset] Loading RNN data...")
        self.rnn_data = np.load(rnn_path)["data"]

        self.total_len          = len(self.labels)
        self.samples_per_subject = self.total_len / 108

        if num_subjects and 0 < num_subjects < 108:
            self.limit_idx = int(num_subjects * self.samples_per_subject)
            print(f"[EEGDataset] Limited to first {num_subjects} subjects ({self.limit_idx:,} samples)")
        else:
            self.limit_idx = self.total_len

    def __len__(self) -> int:
        return self.limit_idx

    def __getitem__(self, idx: int):
        cnn   = self.cnn_data[idx].astype(np.float32)[:, np.newaxis, :, :]
        rnn   = self.rnn_data[idx].astype(np.float32)
        return torch.from_numpy(cnn), torch.from_numpy(rnn), int(self.labels[idx])


class NoisySubset(Dataset):
    def __init__(self, subset: Dataset, noise_std: float = 0.0):
        self.subset    = subset
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        cnn_x, rnn_x, label = self.subset[idx]
        if self.noise_std > 0.0:
            cnn_x = cnn_x + torch.randn_like(cnn_x) * self.noise_std
            rnn_x = rnn_x + torch.randn_like(rnn_x) * self.noise_std
        return cnn_x, rnn_x, label


def build_loaders(
    data_dir:     str,
    num_subjects: int   = 12,
    train_ratio:  float = 0.8,
    batch_size:   int   = BATCH,
    num_workers:  int   = NUM_WORKERS,
    pin_memory:   bool  = True,
    noise_std:    float = NOISE_STD,
    seed:         int   = SEED,
):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    full_dataset        = EEGDataset(data_dir, num_subjects=num_subjects)
    total_len           = len(full_dataset)
    test_subject_count  = 2
    train_val_count     = num_subjects - test_subject_count
    split_idx           = int(train_val_count * full_dataset.samples_per_subject)

    test_set = Subset(full_dataset, list(range(split_idx, total_len)))
    pool_set = Subset(full_dataset, list(range(0, split_idx)))

    n_train = int(train_ratio * len(pool_set))
    n_val   = len(pool_set) - n_train
    train_set, val_set = random_split(pool_set, [n_train, n_val], generator=generator)

    noisy_train = NoisySubset(train_set, noise_std=noise_std)

    _kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(noisy_train, shuffle=True,  **_kw)
    val_loader   = DataLoader(val_set,     shuffle=False, **_kw)
    test_loader  = DataLoader(test_set,    shuffle=False, **_kw)

    noise_tag = f"std={noise_std}" if noise_std > 0 else "disabled"
    print(f"\n[build_loaders] Subject-Independent Split:")
    print(f"  Subjects 01-{train_val_count:02d} (Pool) -> Train: {len(train_set):,} | Val: {len(val_set):,}")
    print(f"  Subjects {train_val_count+1:02d}-{num_subjects:02d} (Blind) -> Test: {len(test_set):,}")
    print(f"  Gaussian Noise: {noise_tag}\n")

    return train_loader, val_loader, test_loader, full_dataset.classes


# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accum_steps: int = 1,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n_samples = 0.0, 0, 0
    amp_device = device.type if device.type != "mps" else "cpu"
    bar = tqdm(loader, desc="  train", leave=False, unit="batch")

    optimiser.zero_grad(set_to_none=True)

    for step, (cnn_x, rnn_x, labels) in enumerate(bar):
        cnn_x  = cnn_x.to(device, dtype=torch.float32, non_blocking=True)
        rnn_x  = rnn_x.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=amp_device):
            logits = model(cnn_x, rnn_x)
            loss   = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        total_loss += loss.item() * accum_steps * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        n_samples  += labels.size(0)

        is_accum_boundary = (step + 1) % accum_steps == 0
        is_last_batch     = (step + 1) == len(loader)

        if is_accum_boundary or is_last_batch:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

        bar.set_postfix(loss=f"{total_loss/n_samples:.4f}", acc=f"{correct/n_samples:.4f}")

    return total_loss / n_samples, correct / n_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, n_samples = 0.0, 0, 0

    for cnn_x, rnn_x, labels in tqdm(loader, desc="   eval", leave=False, unit="batch"):
        cnn_x  = cnn_x.to(device, dtype=torch.float32, non_blocking=True)
        rnn_x  = rnn_x.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits      = model(cnn_x, rnn_x)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        n_samples  += labels.size(0)

    return total_loss / n_samples, correct / n_samples


def run_training(
    model:              nn.Module,
    train_loader:       DataLoader,
    val_loader:         DataLoader,
    device:             torch.device,
    epochs:             int   = EPOCHS,
    lr:                 float = LR,
    output_dir:         str   = OUTPUT_DIR,
    checkpoint_interval: int  = CHECKPOINT_INTERVAL,
    early_stopping_patience: int = EARLY_STOP_PATIENCE,
    accum_steps:        int   = GRAD_ACCUM_STEPS,
) -> Tuple[Dict, float]:
    os.makedirs(output_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler    = GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=LR_T_MAX, eta_min=LR_ETA_MIN
    )

    ckpt_path   = os.path.join(output_dir, "best_parallel.pt")
    resume_path = os.path.join(output_dir, "parallel_resume.pt")

    best_val_acc   = 0.0
    history        = defaultdict(list)
    start_epoch    = 1
    patience_counter = 0

    if os.path.exists(resume_path):
        print(f"[trainer] Resuming from {resume_path} ...")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimiser.load_state_dict(ckpt["optim_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "sched_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["sched_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history      = defaultdict(list, ckpt["history"])

    model.to(device)
    eff_batch = train_loader.batch_size * accum_steps
    print(f"\n[trainer] Training | epochs {start_epoch}→{epochs} | device={device}")
    print(f"[trainer] Grad accumulation: {accum_steps} steps | effective batch: {eff_batch}\n")

    epoch_bar = tqdm(
        range(start_epoch, epochs + 1), desc="Epochs", unit="ep",
        initial=start_epoch - 1, total=epochs
    )

    try:
        for epoch in epoch_bar:
            t0 = time.perf_counter()
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimiser, scaler, device, accum_steps)
            vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            history["tr_loss"].append(tr_loss); history["tr_acc"].append(tr_acc)
            history["vl_loss"].append(vl_loss); history["vl_acc"].append(vl_acc)
            history["lr"].append(current_lr)

            new_best = ""
            if vl_acc > best_val_acc:
                best_val_acc   = vl_acc
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                new_best = " *"
            else:
                patience_counter += 1

            if epoch % checkpoint_interval == 0 or epoch == epochs:
                snap = {"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optim_state_dict": optimiser.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "sched_state_dict": scheduler.state_dict(),
                        "best_val_acc": best_val_acc, "history": dict(history)}
                torch.save(snap, os.path.join(output_dir, f"parallel_epoch_{epoch}.pt"))
                torch.save(snap, resume_path)

                import pandas as pd
                pd.DataFrame(dict(history)).to_csv(
                    os.path.join(output_dir, "parallel_history.csv"), index_label="epoch_idx"
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            elapsed = time.perf_counter() - t0
            epoch_bar.set_postfix(
                tr_acc=f"{tr_acc:.4f}", vl_acc=f"{vl_acc:.4f}",
                best=f"{best_val_acc:.4f}", lr=f"{current_lr:.2e}", s=f"{elapsed:.1f}"
            )
            if new_best:
                tqdm.write(f"  Ep {epoch:3d}/{epochs}{new_best}  lr={current_lr:.2e}  val_acc={vl_acc:.4f}")

            if patience_counter >= early_stopping_patience:
                tqdm.write(f"\nEarly stopping at epoch {epoch} (no improvement in {early_stopping_patience} epochs)")
                break

    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted. Returning current history...")

    return dict(history), best_val_acc


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG Motor Imagery — Parallel CNN-GRU (Compute VM)")
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--batch",       type=int,   default=BATCH)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--subjects",    type=int,   default=12)
    parser.add_argument("--data_dir",    type=str,   default=DATA_DIR)
    parser.add_argument("--output_dir",  type=str,   default=OUTPUT_DIR)
    parser.add_argument("--gcs_bucket",  type=str,   default=GCS_BUCKET)
    parser.add_argument("--gcs_prefix",  type=str,   default=GCS_DATA_PREFIX)
    parser.add_argument("--num_workers", type=int,   default=NUM_WORKERS)
    parser.add_argument("--download",    action="store_true", help="Force re-download from GCS")
    parser.add_argument("--eval_only",   action="store_true", help="Skip training; run blind test only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -- Generate Dynamic Output Dir --
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"results_{args.epochs}_{timestamp}"
    args.output_dir = os.path.join(os.path.dirname(args.output_dir), folder_name)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  EEG Motor Imagery — Parallel CNN-GRU")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Subjects   : {args.subjects}")
    print(f"  Epochs     : {args.epochs}")
    print(f"{'='*60}\n")

    # -- Save Hyperparameters --
    config = {
        "global_params": {
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "subjects": args.subjects,
            "window": WINDOW,
            "noise_std": NOISE_STD,
            "grad_accum": GRAD_ACCUM_STEPS,
        },
        "model_cfg": PARALLEL_CFG
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(f"[main] Hyperparameters saved to {args.output_dir}/config.json")

    # ── 1. Download from GCS if needed ───────────────────────────────────────
    required_files = [
        "S001_S108_win10_labels.npz",
        "S001_S108_win10_cnn_data.npz",
        "S001_S108_win10_rnn_data.npz",
    ]
    data_present = all(
        os.path.exists(os.path.join(args.data_dir, f)) for f in required_files
    )

    if args.download or not data_present:
        print("[main] Downloading data from GCS...")
        download_gcs_prefix(args.gcs_bucket, args.gcs_prefix, args.data_dir)
    else:
        print("[main] Data already present, skipping download.")

    # ── 2. DataLoaders ────────────────────────────────────────────────────────
    pin_memory = device.type == "cuda"
    train_loader, val_loader, test_loader, class_names = build_loaders(
        data_dir=args.data_dir,
        num_subjects=args.subjects,
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = build_model(
        cfg=PARALLEL_CFG, window=WINDOW, n_classes=N_CLASSES, dropout=DROPOUT
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] Model parameters: {n_params:,}\n")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    history, best_val_acc = {}, 0.0

    if not args.eval_only:
        history, best_val_acc = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            output_dir=args.output_dir,
        )

    # ── 5. Final blind test ───────────────────────────────────────────────────
    best_ckpt = os.path.join(args.output_dir, "best_parallel.pt")
    if os.path.exists(best_ckpt):
        print(f"\n[main] Loading best checkpoint from {best_ckpt}")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"  BLIND TEST (Subjects 11-12)")
    print(f"  Loss : {test_loss:.4f}")
    print(f"  Acc  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"  Best Val Acc : {best_val_acc:.4f}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {"test_loss": test_loss, "test_acc": test_acc, "best_val_acc": best_val_acc, "classes": class_names}
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[main] Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
