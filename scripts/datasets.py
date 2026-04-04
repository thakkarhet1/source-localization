"""
datasets.py — PyTorch Dataset and DataLoader builder with 3-way subject-independent split.

Split Logic:
  1. Load all subjects for the pool (e.g. 12 subjects).
  2. Reserve the last 2 subjects (11 & 12) as the 'Blind Test Set' (100% held out).
  3. Split the first 10 subjects into 'Train' and 'Validation' based on TRAIN_RATIO.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import config


class EEGDataset(Dataset):
    """Dataset for the Parallel CNN-GRU model."""

    def __init__(self, data_dir: str, num_subjects: int = 12):
        labels_path = os.path.join(data_dir, "S001_S108_win10_labels.npz")
        cnn_path    = os.path.join(data_dir, "S001_S108_win10_cnn_data.npz")
        rnn_path    = os.path.join(data_dir, "S001_S108_win10_rnn_data.npz")

        if not all(os.path.exists(p) for p in [labels_path, cnn_path, rnn_path]):
            raise FileNotFoundError(f"Missing .npz files in {data_dir}.")

        print(f"[EEGDataset] Loading labels …")
        labels_raw   = np.load(labels_path, allow_pickle=True)["labels"]
        self.classes = sorted(list(set(labels_raw)))
        c2i          = {c: i for i, c in enumerate(self.classes)}
        self.labels  = np.array([c2i[l] for l in labels_raw], dtype=np.int64)

        print(f"[EEGDataset] Loading CNN data …")
        self.cnn_data = np.load(cnn_path)["data"]  # float16

        print(f"[EEGDataset] Loading RNN data …")
        self.rnn_data = np.load(rnn_path)["data"]  # float16

        self.total_len = len(self.labels)
        
        # Calculate subject boundaries (approx 11,384 samples per subject)
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
        label = int(self.labels[idx])
        return torch.from_numpy(cnn), torch.from_numpy(rnn), label


def build_loaders(
    data_dir:     str   = config.PARALLEL_DATA_DIR,
    num_subjects: int   = 12,    # Load 12 subjects (1-10=Train/Val, 11-12=Test)
    train_ratio:  float = 0.8,   # Of the first 10 subjects, 80% go to Train
    batch_size:   int   = config.BATCH,
    num_workers:  int   = config.NUM_WORKERS,
    pin_memory:   bool  = config.PIN_MEMORY,
    seed:         int   = config.SEED,
):
    """Build Train / Val / Blind Test loaders."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    full_dataset = EEGDataset(data_dir, num_subjects=num_subjects)
    total_len    = len(full_dataset)
    
    # Calculate the boundary for the 'Blind Test' (last 2 subjects)
    # We assume subjects are ordered S001...S108
    test_subject_count = 2
    train_val_subject_count = num_subjects - test_subject_count
    
    split_idx = int(train_val_subject_count * full_dataset.samples_per_subject)
    
    # 1. Slice out the Blind Test Set (Subjects 11-12)
    test_indices = list(range(split_idx, total_len))
    test_set     = Subset(full_dataset, test_indices)
    
    # 2. Slice out the Pool for Train/Val (Subjects 1-10)
    pool_indices = list(range(0, split_idx))
    pool_set     = Subset(full_dataset, pool_indices)
    
    # 3. Randomly split the pool into Train and Val
    n_train = int(train_ratio * len(pool_set))
    n_val   = len(pool_set) - n_train
    train_set, val_set = random_split(pool_set, [n_train, n_val], generator=generator)

    _loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        **({"prefetch_factor": config.PREFETCH_FACTOR} if num_workers > 0 else {}),
    )

    train_loader = DataLoader(train_set, shuffle=True,  **_loader_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, **_loader_kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False, **_loader_kwargs)

    print(f"\n[build_loaders] Subject-Independent Split Summary:")
    print(f"  Subjects 01-10 (Pool) -> Train: {len(train_set):,} | Val: {len(val_set):,}")
    print(f"  Subjects 11-12 (Blind)-> Test : {len(test_set):,}")
    print(f"  Total Active Samples  : {total_len:,}\n")

    return train_loader, val_loader, test_loader, full_dataset.classes
