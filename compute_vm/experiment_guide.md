# EEG Motor Imagery: Experimentation Guide

This document explains the workflow for iterating on the Parallel CNN-GRU model using your local machine and the Google Cloud Compute VM.

## 1. Modifying the Model & Hyperparameters

Most changes happen at the top of `compute_vm/train.py`.

### Architectural Changes
Look for the `PARALLEL_CFG` dictionary (around line 50). This is where you adjust:
- `gru_hidden`: The "memory" capacity of the temporal branch.
- `fusion`: How spatial and temporal features are combined (`"add"`, `"concat"`, `"concat_fc"`).

### Hyperparameters
Default values are set in the global constants (lines 35-50):
- `EPOCHS`: How many times to see the full dataset.
- `BATCH`: How many samples to process at once.
- `LR`: The learning rate (speed of learning).

---

## 2. Syncing Your Changes to the VM

After saving your changes locally, push the file to your VM using the `gcloud` CLI:

```bash
gcloud compute scp compute_vm/train.py instance-20260506-115403:~/source-localization/compute_vm/train.py --zone=us-central1-a
```

---

## 3. Running the Experiment

### Step A: Connect to the VM
```bash
gcloud compute ssh instance-20260506-115403 --zone=us-central1-a
```

### Step B: Manage the Environment
Always activate the virtual environment before running the script:
```bash
cd ~/source-localization/compute_vm
source venv/bin/activate
```

### Step C: Launch with `nohup` (Recommended)
`nohup` allows the script to keep running even if you disconnect from the VM.
```bash
nohup python train.py \
  --epochs 10 \
  --data_dir ~/eeg_data \
  --output_dir ~/eeg_results \
  > ~/training.log 2>&1 &
```
*   `> ~/training.log 2>&1`: Redirects both standard output and errors to a log file.
*   `&`: Puts the process in the background.

---

## 4. Monitoring & Managing Processes

### Watch Training in Real-Time
In a new terminal window:
```bash
gcloud compute ssh instance-20260506-115403 --zone=us-central1-a --command="tail -f ~/training.log"
```

### Stop a Running Experiment
If you want to stop the current run to change something:
1. Find the Process ID (PID):
   ```bash
   ps aux | grep train.py
   ```
2. Kill the process:
   ```bash
   kill <PID_NUMBER>
   ```

---

## 5. Retrieving Results

Once training is finished, your results (best model weights, CSV history, and JSON test results) will be in `~/eeg_results`.

Download them to your local machine:
```bash
gcloud compute scp --recurse instance-20260506-115403:~/eeg_results ./local_results --zone=us-central1-a
```

## 6. Starting and Stopping the VM

**Always stop the VM when not training** — you're billed for every hour it runs.

### Start the VM
```bash
gcloud compute instances start instance-20260506-115403 --zone=us-central1-a
```

### Stop the VM
```bash
gcloud compute instances stop instance-20260506-115403 --zone=us-central1-a
```

### Check VM status
```bash
gcloud compute instances list
```

---

## Pro Tips
- **Disk Space**: The dataset is ~5GB. If you run out of space, use `df -h /` to check.
- **GPU**: Your current VM is CPU-only. If you add a GPU later, the script will automatically detect it and run much faster.
- **Screen**: For a more interactive persistent session, use `screen -S training`. (Detach with `Ctrl+A, D`; Reattach with `screen -r training`).
