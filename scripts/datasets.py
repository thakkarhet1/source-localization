"""
datasets.py — PyTorch Dataset and DataLoader builder.

All data is loaded fully into memory in __init__ for maximum speed on M2 Pro
(no per-item file I/O overhead).  float16 → float32 casting is deferred to
__getitem__ so the in-memory footprint stays halved while the GPU always
receives float32 tensors.

num_workers=0 (default from config) is the safe choice on macOS where
fork-based multiprocessing can deadlock with MPS/OpenMP.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import config


class EEGDataset(Dataset):
    """Dataset for the Parallel CNN-GRU model.

    Loads all data from the main .npz files eagerly at construction time.

    Expected files in data_dir:
        S001_S108_win10_cnn_data.npz
        S001_S108_win10_rnn_data.npz
        S001_S108_win10_labels.npz

    __getitem__ returns: (cnn_tensor, rnn_tensor, label).
    """

    def __init__(self, data_dir: str, num_subjects: int = None):
        labels_path = os.path.join(data_dir, "S001_S108_win10_labels.npz")
        cnn_path    = os.path.join(data_dir, "S001_S108_win10_cnn_data.npz")
        rnn_path    = os.path.join(data_dir, "S001_S108_win10_rnn_data.npz")

        if not all(os.path.exists(p) for p in [labels_path, cnn_path, rnn_path]):
            raise FileNotFoundError(
                f"One or more data files missing in {data_dir}. "
                "Expected: labels, cnn, and rnn .npz files."
            )

        # ── Load labels ──────────────────────────────────────────────────────
        print(f"[EEGDataset] Loading labels from {os.path.basename(labels_path)} …")
        labels_raw   = np.load(labels_path, allow_pickle=True)["labels"]
        self.classes = sorted(list(set(labels_raw)))
        c2i          = {c: i for i, c in enumerate(self.classes)}
        self.labels  = np.array([c2i[l] for l in labels_raw], dtype=np.int64)

        # ── Load CNN data (fully into RAM) ───────────────────────────────────
        print(f"[EEGDataset] Loading CNN data: {os.path.basename(cnn_path)} …")
        self.cnn_data = np.load(cnn_path)["data"]          # kept as float16 to save RAM

        # ── Load RNN data (fully into RAM) ───────────────────────────────────
        print(f"[EEGDataset] Loading RNN data: {os.path.basename(rnn_path)} …")
        self.rnn_data = np.load(rnn_path)["data"]          # kept as float16 to save RAM

        self.total_len = len(self.labels)

        # ── Optional subject sub-sampling ────────────────────────────────────
        if num_subjects and 0 < num_subjects < 108:
            self.total_len = int((num_subjects / 108) * self.total_len)
            print(
                f"[EEGDataset] Limiting to first {num_subjects} subjects "
                f"(~{self.total_len:,} samples)"
            )

        print(
            f"[EEGDataset] Ready — {self.total_len:,} samples | "
            f"classes = {self.classes}"
        )

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int):
        # Cast float16 → float32 here (GPU needs float32; storing float16 halves RAM)
        cnn   = self.cnn_data[idx].astype(np.float32)[:, np.newaxis, :, :]  # (W, 1, H, Wd)
        rnn   = self.rnn_data[idx].astype(np.float32)                        # (W, n_elec)
        label = int(self.labels[idx])
        return torch.from_numpy(cnn), torch.from_numpy(rnn), label


def build_loaders(
    data_dir:    str   = config.PARALLEL_DATA_DIR,
    num_subjects: int  = config.NUM_SUBJECTS,
    train_ratio: float = config.TRAIN_RATIO,
    batch_size:  int   = config.BATCH,
    num_workers: int   = config.NUM_WORKERS,
    pin_memory:  bool  = config.PIN_MEMORY,
    seed:        int   = config.SEED,
):
    """Build train / test DataLoaders from the parallel .npz data directory.

    Args:
        data_dir     : absolute path to directory containing the .npz files
        num_subjects : number of subjects to include (None/0 for all 108)
        train_ratio  : fraction reserved for training
        batch_size   : mini-batch size
        num_workers  : DataLoader workers (0 = main process, safe on macOS)
        pin_memory   : pin host memory for faster CUDA transfer (False on MPS/CPU)
        seed         : random seed for reproducible split

    Returns:
        (train_loader, test_loader, class_names)
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    dataset = EEGDataset(data_dir, num_subjects=num_subjects)

    n_train = int(train_ratio * len(dataset))
    n_test  = len(dataset) - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)

    # prefetch_factor is only valid when num_workers > 0
    _kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        **({"prefetch_factor": config.PREFETCH_FACTOR} if num_workers > 0 else {}),
    )

    train_loader = DataLoader(train_set, shuffle=True,  **_kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False, **_kwargs)

    print(
        f"[build_loaders] train={n_train:,} | test={n_test:,} | "
        f"batches/epoch={len(train_loader)}"
    )
    return train_loader, test_loader, dataset.classes
