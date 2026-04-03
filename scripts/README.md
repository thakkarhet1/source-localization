# EEG Motor Imagery — `scripts/` folder

Converted from `eeg_models_colab.ipynb` and optimised for **Apple M2 8 GB RAM**.

---

## File overview

| File | Role |
|---|---|
| `config.py` | **Single source of truth** for all hyperparameters, paths, and device selection |
| `models.py` | `CascadeCNNRNN` and `ParallelCNNRNN` architecture definitions + `build_model()` factory |
| `datasets.py` | `CascadeDataset`, `ParallelDataset` (memory-mapped) + `build_loaders()` factory |
| `trainer.py` | `train_epoch()`, `evaluate()`, and `run_training()` with resume checkpointing |
| `evaluate_and_plot.py` | Final evaluation, confusion matrix, per-class accuracy, training curves |
| `sweep.py` | Short hyperparameter sweep over fusion strategies / model sizes |
| `main.py` | **Entry point** — wires everything together with a CLI |

---

## Quick start

```bash
# From repo root
cd /path/to/Cascade-Parallel

# Train the parallel model (default)
python scripts/main.py

# Train cascade model
python scripts/main.py --model cascade

# Evaluate only (load best checkpoint, no training)
python scripts/main.py --model parallel --eval-only

# Train + optional sweep
python scripts/main.py --model parallel --sweep

# Override any config value at the CLI
python scripts/main.py --model parallel --epochs 30 --batch 64 --lr 3e-4
```

---

## M2 8 GB RAM optimisations

| Change | Reason |
|---|---|
| **MPS device** (`torch.backends.mps.is_available()`) | Uses Apple Silicon GPU cores instead of CPU |
| **Batch size 128** (was 512 on T4) | T4 has 16 GB VRAM; M2 shares 8 GB with the OS |
| **`fc_size` / `lstm_hidden` = 256** (was 512–1024) | ~4× fewer parameters → ~4× less activation memory |
| **`num_workers = 0`** | macOS fork-based multiprocessing can deadlock with MPS |
| **`pin_memory = False`** on MPS | `pin_memory` only helps CUDA; wastes RAM on MPS |
| **`torch.mps.empty_cache()`** after each epoch | Returns Metal heap pages to the OS |
| **`mmap_mode='r'` numpy loading** | Dataset stays on disk; RAM usage is proportional to batch size only |
| **`zero_grad(set_to_none=True)`** | Slightly faster and uses less memory than zeroing tensors |

### Monitoring memory on macOS

Open **Activity Monitor → Memory tab** and watch:
- *Memory Used* — should stay below ~6 GB during training  
- *Swap Used* — ideally 0; swap will slow training significantly

If you see swap, reduce `BATCH` or the model dimensions in `config.py`.

---

## Checkpoint layout (`results/`)

```
results/
  best_parallel.pt          ← best test-accuracy weights only (small, fast to load)
  parallel_resume.pt        ← latest full checkpoint (model + optimiser + history)
  parallel_epoch_10.pt      ← snapshot at epoch 10
  parallel_epoch_20.pt      ← …and so on every 10 epochs
  parallel_curves.png
  parallel_confusion.png
  parallel_per_class.png
  parallel_sweep.png        ← only if --sweep was used
```
