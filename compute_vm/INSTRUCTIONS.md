# Running EEG Training on a GCP Compute VM

## Prerequisites

- A GCP Compute VM with a GPU (e.g. `n1-standard-8` + `nvidia-tesla-t4` or better)
- Data uploaded to GCS bucket `parallel_eeg_decoding` under `eeg_data/npz_4class_parallel_W10/`
- VM has Python 3.9+ and CUDA drivers installed

---

## 1. SSH into your VM

```bash
gcloud compute ssh <your-vm-name> --zone <your-zone>
```

---

## 2. Authenticate with GCP (first time only)

The VM's service account needs Storage Object Viewer access to the bucket, or run:

```bash
gcloud auth application-default login
```

---

## 3. Clone the repo and navigate to the script

```bash
git clone <your-repo-url>
cd source-localization/compute_vm
```

---

## 4. Set up a Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **Check CUDA version:** `nvcc --version` or `nvidia-smi`
> For CUDA 11.8: use `cu118` instead of `cu121`.

---

## 5. Run training

### Full run (downloads data on first run, then trains):

```bash
python train.py
```

### With custom arguments:

```bash
python train.py \
  --epochs 100 \
  --batch 32 \
  --lr 1e-4 \
  --subjects 12 \
  --data_dir /home/jupyter/eeg_data/npz_4class_parallel_W10 \
  --output_dir /home/jupyter/eeg_results \
  --num_workers 4
```

### Force re-download data from GCS:

```bash
python train.py --download
```

### Skip training, run blind test only (requires a saved checkpoint):

```bash
python train.py --eval_only
```

---

## 6. Run in a persistent session (survives SSH disconnect)

```bash
# Using screen
screen -S training
source venv/bin/activate
python train.py 2>&1 | tee training.log
# Detach: Ctrl+A, D  |  Reattach: screen -r training

# Or using nohup
nohup python train.py > training.log 2>&1 &
tail -f training.log
```

---

## 7. Monitor GPU usage

```bash
watch -n 2 nvidia-smi
```

---

## 8. Output files

| File | Description |
|------|-------------|
| `eeg_results/best_parallel.pt` | Best model weights (by validation accuracy) |
| `eeg_results/parallel_resume.pt` | Full checkpoint for resuming training |
| `eeg_results/parallel_epoch_N.pt` | Periodic snapshots every 10 epochs |
| `eeg_results/parallel_history.csv` | Per-epoch loss/accuracy/LR log |
| `eeg_results/results.json` | Final blind test results |

---

## 9. Resuming interrupted training

Training resumes automatically if `eeg_results/parallel_resume.pt` exists. Just re-run:

```bash
python train.py
```

---

## 10. Copying results back to your local machine

```bash
# From your local machine
gcloud compute scp --recurse <your-vm-name>:/home/jupyter/eeg_results ./results --zone <your-zone>
```

Or upload to GCS:

```bash
# On the VM
gsutil -m cp -r /home/jupyter/eeg_results gs://parallel_eeg_decoding/results/
```

---

## GCS bucket layout expected

```
gs://parallel_eeg_decoding/
  eeg_data/npz_4class_parallel_W10/
    S001_S108_win10_labels.npz
    S001_S108_win10_cnn_data.npz
    S001_S108_win10_rnn_data.npz
```

---

## Key hyperparameters (edit top of `train.py` or pass as CLI args)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--epochs` | 100 | Max training epochs (early stopping at 15 without improvement) |
| `--batch` | 32 | Per-GPU batch size; effective batch = 32 × 4 accum steps = 128 |
| `--lr` | 1e-4 | Initial LR; cosine-annealed to 1e-6 |
| `--subjects` | 12 | Total subjects loaded (10 train/val + 2 blind test) |
| `--num_workers` | 4 | DataLoader workers; set to 0 if you see multiprocessing errors |
