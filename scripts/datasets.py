"""
datasets.py — Memory-efficient PyTorch Dataset and DataLoader builder.

Design goals for M2 8 GB RAM:
  • float32 casting happens per-item (inside __getitem__) rather than
    loading an entire float32 copy of the dataset upfront.
  • num_workers=0 (default from config) is the safe choice on macOS
    where fork-based multiprocessing can deadlock with MPS/OpenMP.
"""

import os
import bisect
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import config


class LazyEEGDataset(Dataset):
    """Dataset for the Parallel CNN-RNN model.

    Loads data directly from the main .npz files.

    Expected files in data_dir:
        S001_S108_win10_cnn_data.npz
        S001_S108_win10_rnn_data.npz
        S001_S108_win10_labels.npz

    __getitem__ returns: (cnn_tensor, rnn_tensor, label).
    """

    def __init__(self, data_dir: str, num_subjects: int = None):
        labels_path = os.path.join(data_dir, "S001_S108_win10_labels.npz")
        cnn_path = os.path.join(data_dir, "S001_S108_win10_cnn_data.npz")
        rnn_path = os.path.join(data_dir, "S001_S108_win10_rnn_data.npz")

        if not all(os.path.exists(p) for p in [labels_path, cnn_path, rnn_path]):
            raise FileNotFoundError(f"One or more data files missing in {data_dir}. Expected: labels, cnn, and rnn .npz files.")

        print(f"[LazyEEGDataset] Loading labels from {os.path.basename(labels_path)}...")
        labels_raw = np.load(labels_path, allow_pickle=True)["labels"]
        self.classes = sorted(list(set(labels_raw)))
        c2i = {c: i for i, c in enumerate(self.classes)}
        self.labels = np.array([c2i[l] for l in labels_raw], dtype=np.int64)
        
        print(f"[LazyEEGDataset] Loading CNN data: {os.path.basename(cnn_path)}")
        self.cnn_data = np.load(cnn_path)["data"]
        
        print(f"[LazyEEGDataset] Loading RNN data: {os.path.basename(rnn_path)}")
        self.rnn_data = np.load(rnn_path)["data"]
        
        self.total_len = len(self.labels)
        if num_subjects and 0 < num_subjects < 108:
            # Approximate subject-based limiting
            self.total_len = int((num_subjects / 108) * self.total_len)
            print(f"[LazyEEGDataset] Limiting to first {num_subjects} subjects (~{self.total_len:,} samples)")

        print(f"[LazyEEGDataset] Total samples = {self.total_len} | classes = {self.classes}")

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int):
        if idx >= self.total_len:
            raise IndexError("Index out of bounds")
        
        # Slicing arrays is very fast; no file re-opening overhead
        cnn = self.cnn_data[idx].astype(np.float16)[:, np.newaxis, :, :]  # (W, 1, H, Wd)
        rnn = self.rnn_data[idx].astype(np.float16)                       # (W, n_elec)
        label = self.labels[idx]
        
        return torch.from_numpy(cnn), torch.from_numpy(rnn), int(label)


def build_loaders(
    data_dir: str = config.PARALLEL_DATA_DIR,
    num_subjects: int = config.NUM_SUBJECTS,
    train_ratio: float = config.TRAIN_RATIO,
    batch_size: int = config.BATCH,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = config.PIN_MEMORY,
    prefetch_factor: int = config.PREFETCH_FACTOR,
    seed: int = config.SEED,
):
    """Build train / test DataLoaders from the parallel .npz data directory.

    Args:
        data_dir        : path to directory containing the .npz files
        num_subjects    : number of subjects to include (None/0 for all 108)
        train_ratio     : fraction reserved for training
        batch_size      : mini-batch size
        num_workers     : DataLoader workers (0 = main process, safe on macOS)
        pin_memory      : pin host memory for faster CUDA transfer (False on MPS/CPU)
        prefetch_factor : number of batches loaded in advance by each worker
        seed            : random seed for reproducible split

    Returns:
        (train_loader, test_loader, class_names)
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    dataset = LazyEEGDataset(data_dir, num_subjects=num_subjects)

    n_train = int(train_ratio * len(dataset))
    n_test  = len(dataset) - n_train
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)

    _kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_set, shuffle=True,  **_kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False, **_kwargs)

    print(
        f"[build_loaders] train={n_train:,} | test={n_test:,} | "
        f"batches/epoch={len(train_loader)}"
    )
    return train_loader, test_loader, dataset.classes
