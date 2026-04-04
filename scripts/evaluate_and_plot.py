"""
evaluate_and_plot.py — Final evaluation and result visualisation.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

import config
from trainer import evaluate


def plot_training_curves(
    history: Dict[str, List[float]],
    best_acc: float,
    output_dir: str = config.OUTPUT_DIR,
) -> None:
    if history is None:
        return
        
    n_epochs = len(history["tr_loss"])
    ep       = range(1, n_epochs + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Parallel CNN-GRU — Training Curves (Train vs Val)", fontweight="bold")

    # Change keys from te_* to vl_* to match trainer.py
    ax_loss.plot(ep, history["tr_loss"], label="Train", color="steelblue", lw=1.5)
    ax_loss.plot(ep, history["vl_loss"], label="Val",   color="orangered",  lw=1.5)
    ax_loss.set(xlabel="Epoch", ylabel="Loss", title="Cross-Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(ep, [a * 100 for a in history["tr_acc"]], label="Train", color="steelblue", lw=1.5)
    ax_acc.plot(ep, [a * 100 for a in history["vl_acc"]], label="Val",   color="orangered",  lw=1.5)
    ax_acc.axhline(
        best_acc * 100, ls="--", color="green", alpha=0.7,
        label=f"Best Val {best_acc * 100:.2f}%",
    )
    ax_acc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax_acc.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "parallel_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Training curves saved → {out_path}")
    plt.close(fig)


def plot_confusion_matrix(
    true_labels: List[int],
    predictions: List[int],
    class_names: List[str],
    output_dir: str = config.OUTPUT_DIR,
) -> np.ndarray:
    cm      = confusion_matrix(true_labels, predictions)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Independent Test Set (Subjects 11-12) — Confusion Matrix", fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Normalised"],
    ):
        sns.heatmap(
            data, ax=ax, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.4,
        )
        ax.set(xlabel="Predicted", ylabel="True", title=title)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "parallel_confusion.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Confusion matrix saved → {out_path}")
    plt.close(fig)

    return cm_norm


def plot_per_class_accuracy(
    cm_norm: np.ndarray,
    class_names: List[str],
    overall_acc: float,
    output_dir: str = config.OUTPUT_DIR,
) -> None:
    per_class = cm_norm.diagonal()
    palette   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"] + ["#8172B2"] * 10
    colors    = palette[: len(class_names)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, per_class * 100, color=colors, edgecolor="white", linewidth=0.8)

    for bar, v in zip(bars, per_class):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{v * 100:.1f}%",
            ha="center", fontsize=10, fontweight="bold",
        )

    ax.axhline(
        overall_acc * 100, ls="--", color="black", alpha=0.5,
        label=f"Overall Test {overall_acc * 100:.1f}%",
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set(ylabel="Accuracy (%)", title="Independent Test Accuracy Per Class", ylim=(0, 115))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "parallel_per_class.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] Per-class accuracy saved → {out_path}")
    plt.close(fig)


def run_final_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    history: Dict[str, List[float]],
    best_acc: float,
    device: torch.device = config.DEVICE,
    output_dir: str = config.OUTPUT_DIR,
    ckpt_path: str | None = None,
) -> None:
    if ckpt_path is None:
        ckpt_path = os.path.join(output_dir, "best_parallel.pt")

    print(f"\n[evaluate] Loading best weights from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    _, final_acc, preds, true_labels = evaluate(model, test_loader, criterion, device)

    print(f"\n🏆  FINAL INDEPENDENT TEST ACCURACY: {final_acc:.4f}  ({final_acc * 100:.2f}%)")
    print("\n── Classification Report ──")
    print(classification_report(true_labels, preds, target_names=class_names))

    plot_training_curves(history, best_acc, output_dir)
    cm_norm = plot_confusion_matrix(true_labels, preds, class_names, output_dir)
    plot_per_class_accuracy(cm_norm, class_names, final_acc, output_dir)

    print("\n✅  All evaluation plots updated for Subject-Independent split.")
