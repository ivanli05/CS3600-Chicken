# AgentPro Training on PACE ICE Cluster

Complete step-by-step guide for training AgentPro's neural network evaluator on Georgia Tech's PACE ICE cluster with H100 GPUs.

---

## Prerequisites

1. **PACE Account**: Ensure you have access to PACE ICE cluster
2. **SSH Access**: Set up SSH keys for passwordless login (recommended)
3. **Storage**: Ensure you have enough space in your home directory (~500MB for code + data)

---

## Step 1: Connect to PACE

```bash
# SSH into PACE login node
ssh <your-gt-username>@login-ice.pace.gatech.edu

# You should see the PACE welcome message
```

---

## Step 2: Transfer Code to PACE

### Option A: Using Git (Recommended)

```bash
# On PACE login node
cd ~
mkdir -p projects
cd projects

# Clone your repository
git clone https://github.com/your-username/CS3600-Chicken.git
cd CS3600-Chicken/dist/3600-agents/AgentPro/training
```

### Option B: Using SCP

```bash
# From your local machine (NOT on PACE)
cd /Users/ivanli/Documents/GitHub/CS3600-Chicken/dist/3600-agents

# Transfer the entire AgentPro directory
scp -r AgentPro <your-gt-username>@login-ice.pace.gatech.edu:~/projects/
```

**Verify the transfer:**
```bash
# On PACE
cd ~/projects/CS3600-Chicken/dist/3600-agents/AgentPro/training
ls -la

# You should see:
# - generate_data_job.sbatch
# - train_job.sbatch
# - generate_data_parallel.py
# - train_on_gpu.py
# - config.yaml
# - monitor_training.py
# - README_GPU_TRAINING.md
# - PACE_INSTRUCTIONS.md

# Also verify the engine directory exists:
ls ~/projects/CS3600-Chicken/dist/engine/game
# You should see Board.py, enums.py, etc.
```

---

## Step 3: Set Up Directory Structure

```bash
# Navigate to training directory
cd ~/projects/CS3600-Chicken/dist/3600-agents/AgentPro/training

# Create logs directory
mkdir -p logs

# Verify the requirements.txt exists in the dist directory
ls -la ../../../requirements.txt

# Verify the game engine exists
ls -la ../../../../engine/game

# Your directory structure should look like:
# ~/projects/CS3600-Chicken/
#   ├── dist/
#   │   ├── requirements.txt
#   │   ├── engine/
#   │   │   └── game/
#   │   └── 3600-agents/
#   │       └── AgentPro/
#   │           └── training/  ← You are here
```

---

## Step 4: Verify Module Availability

```bash
# Check available Python modules
module avail python

# You should see python/3.10 or similar
# If not available, check what's available and modify the sbatch files accordingly

# Check available GPU partitions
sinfo -p gpu-h100

# This will show you the H100 GPU nodes and their availability
```

---

## Step 5: Submit Data Generation Job

```bash
# Make sure you're in the training directory
cd ~/projects/CS3600-Chicken/dist/3600-agents/AgentPro/training

# Submit the data generation job
sbatch generate_data_job.sbatch

# Note the job ID (e.g., 12345678)
# You should see output like:
# Submitted batch job 12345678
```

**Monitor the data generation:**

```bash
# Check job status
squeue -u $USER

# Watch the output log in real-time
tail -f logs/datagen-<jobid>.out

# Example:
# tail -f logs/datagen-12345678.out

# You should see progress like:
# Progress: 5000/50000 (10.0%) | Rate: 12.3 pos/s | ETA: 60.7 min
```

**Expected time**: 2-3 hours for 50,000 positions with 32 CPU cores

---

## Step 6: Verify Data Generation Completed

```bash
# Check if the data file was created
ls -lh training_data.json

# You should see a file around 50-100 MB
# Example output:
# -rw-r--r-- 1 username group 87M Nov 18 15:23 training_data.json

# Check the completion message in the log
tail -20 logs/datagen-<jobid>.out

# You should see:
# ✓ Dataset saved!
# Generated file: training_data.json (87M)
# Ready for training! Submit train_job.sbatch
```

---

## Step 7: Submit Training Job

```bash
# Submit the training job
sbatch train_job.sbatch

# Note the job ID
```

**Monitor the training:**

```bash
# Check job status
squeue -u $USER

# Watch GPU usage (while job is running)
srun --jobid=<your-job-id> nvidia-smi

# Monitor training progress
tail -f logs/train-<jobid>.out

# Or use the monitoring script
python monitor_training.py --log-file logs/train-<jobid>.out
```

**Expected time**: 4-6 hours for 200 epochs on H100 GPU

**What you should see in the logs:**

```
==========================================
Job started at: Mon Nov 18 15:30:00 EST 2024
Job ID: 12345679
Node: gpu-node-05
==========================================

GPU Information:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.1   |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA H100         On   | 00000000:00:1E.0 Off |                    0 |
| N/A   32C    P0    50W / 350W |      0MiB / 81559MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

Creating virtual environment...
Installing dependencies...
...
Starting training...

Epoch 1/200 | Train Loss: 0.1234 | Val Loss: 0.1456 | LR: 0.001000
  ✓ Saved new best model (val_loss=0.1456)
Epoch 2/200 | Train Loss: 0.0987 | Val Loss: 0.1123 | LR: 0.001000
  ✓ Saved new best model (val_loss=0.1123)
...
```

---

## Step 8: Monitor Training Progress

### Method 1: Real-time Log Monitoring

```bash
# Simple tail
tail -f logs/train-<jobid>.out

# With color highlighting (if available)
tail -f logs/train-<jobid>.out | grep --color=auto -E 'Epoch|✓|Error|$'
```

### Method 2: Using the Monitoring Script

```bash
# One-time snapshot
python monitor_training.py --log-file logs/train-<jobid>.out --once

# Continuous monitoring (updates every 30 seconds)
python monitor_training.py --log-file logs/train-<jobid>.out
```

### Method 3: Check Job Status

```bash
# View all your jobs
squeue -u $USER

# Detailed info about a specific job
scontrol show job <jobid>

# Check if job completed
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed
```

---

## Step 9: Troubleshooting Common Issues

### Issue 1: Job Pending for Long Time

```bash
# Check why job is pending
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %.20R"

# Common reasons:
# - Resources (shows: Resources)
# - Priority (shows: Priority)
# - Partition limit (shows: PartitionResourceLimit)

# Solution: Wait, or adjust resource requirements in sbatch file
```

### Issue 2: Job Failed Immediately

```bash
# Check the error log
cat logs/train-<jobid>.out

# Common issues:
# - Module not found: Check module load commands in sbatch file
# - File not found: Verify paths are correct
# - Permission denied: Check file permissions

# Fix and resubmit:
nano train_job.sbatch  # Edit the file
sbatch train_job.sbatch  # Resubmit
```

### Issue 3: Out of Memory

```bash
# If you see "CUDA out of memory" in logs

# Edit config.yaml to reduce batch size
nano config.yaml

# Change:
# training:
#   batch_size: 128  # Reduce from 256

# Resubmit the job
sbatch train_job.sbatch
```

### Issue 4: Training Too Slow

```bash
# Check GPU utilization
srun --jobid=<jobid> nvidia-smi

# If GPU usage < 80%, try:
# 1. Increase batch_size in config.yaml
# 2. Increase num_workers in config.yaml

# Edit config.yaml:
# training:
#   batch_size: 512  # Increase
# hardware:
#   num_workers: 16  # Increase
```

### Issue 5: Python Module Import Errors

```bash
# If you see "ModuleNotFoundError: No module named 'game'"

# This should be FIXED in the latest version of the code.
# The import paths have been updated to work correctly.

# Verify your directory structure matches this:
cd ~/projects
ls CS3600-Chicken/dist/engine/game  # Should exist
ls CS3600-Chicken/dist/3600-agents/AgentPro  # Should exist

# If you cloned to a different location, the paths should still work
# because they use relative path resolution from the script location.

# If you still get import errors, verify you're running from the correct directory:
cd ~/projects/CS3600-Chicken/dist/3600-agents/AgentPro/training
pwd  # Should end with /AgentPro/training
```

---

## Step 10: Download Trained Model

Once training completes successfully:

```bash
# On PACE, verify the model exists
ls -lh ~/projects/AgentPro/training/best_evaluator.pth

# You should see a file ~1-5 MB
```

### Download to Local Machine

```bash
# From your local machine (NOT on PACE)
# Download the trained model
scp <your-gt-username>@login-ice.pace.gatech.edu:~/projects/AgentPro/training/best_evaluator.pth \
    /Users/ivanli/Documents/GitHub/CS3600-Chicken/dist/3600-agents/AgentPro/

# Download logs for analysis
scp <your-gt-username>@login-ice.pace.gatech.edu:~/projects/AgentPro/training/logs/train-*.out \
    /Users/ivanli/Documents/GitHub/CS3600-Chicken/dist/3600-agents/AgentPro/training/logs/

# Download the training data (if needed)
scp <your-gt-username>@login-ice.pace.gatech.edu:~/projects/AgentPro/training/training_data.json \
    /Users/ivanli/Documents/GitHub/CS3600-Chicken/dist/3600-agents/AgentPro/training/
```

---

## Step 11: Test the Trained Model Locally

```bash
# On your local machine
cd /Users/ivanli/Documents/GitHub/CS3600-Chicken/dist/3600-agents/AgentPro

# Verify the model file is in place
ls -lh best_evaluator.pth

# Test the agent
cd ../../../
python engine/run_local_agents.py AgentPro AgentPro

# Look for this message in the output:
# ✓ Loaded trained evaluator weights from best_evaluator.pth

# The agent should now use the trained neural network!
```

---

## Step 12: Clean Up (Optional)

```bash
# On PACE, once you've downloaded everything

# Remove large files to save space
cd ~/projects/AgentPro/training
rm training_data.json  # ~100 MB
rm -rf datagen_env/    # ~500 MB
rm -rf training_env/   # ~500 MB

# Keep these files:
# - best_evaluator.pth (your trained model)
# - logs/ (for reference)
# - config.yaml (your settings)

# Or remove everything if done
cd ~/projects
rm -rf AgentPro
```

---

## Quick Reference Commands

```bash
# ===== JOB SUBMISSION =====
sbatch generate_data_job.sbatch  # Submit data generation
sbatch train_job.sbatch           # Submit training

# ===== MONITORING =====
squeue -u $USER                   # Check your jobs
tail -f logs/train-<jobid>.out   # Watch training log
srun --jobid=<id> nvidia-smi     # Check GPU usage
python monitor_training.py        # Fancy monitoring

# ===== JOB MANAGEMENT =====
scancel <jobid>                   # Cancel a job
scancel -u $USER                  # Cancel all your jobs
scontrol show job <jobid>         # Detailed job info
sacct -j <jobid>                  # Job accounting info

# ===== FILE TRANSFER =====
# Upload (from local)
scp file.txt $USER@login-ice.pace.gatech.edu:~/path/

# Download (from local)
scp $USER@login-ice.pace.gatech.edu:~/path/file.txt ./

# ===== DEBUGGING =====
cat logs/train-<jobid>.out       # View full log
module avail                      # Check available modules
sinfo -p gpu-h100                # Check GPU availability
```

---

## Expected Results

After successful training (6 hours total):

**Training Metrics:**
- Final Train Loss: 0.025-0.035
- Final Val Loss: 0.030-0.040
- Test Loss: 0.032-0.042

**File Sizes:**
- `training_data.json`: ~50-100 MB
- `best_evaluator.pth`: ~1-5 MB
- Log files: ~1-5 MB each

**Performance:**
- Inference speed: ~1000 evaluations/second
- Should play significantly better than heuristic-only agent
- Win rate vs random: >90%

---

## Getting Help

If you encounter issues:

1. **Check PACE documentation**: https://docs.pace.gatech.edu/
2. **PACE support**: pace-support@oit.gatech.edu
3. **Check job logs**: `logs/train-<jobid>.out` and `logs/datagen-<jobid>.out`
4. **Verify module availability**: `module avail`
5. **Check disk quota**: `quota -s`

---

## Summary Workflow

```
┌─────────────────────────────────────────┐
│ 1. SSH to PACE                          │
│    ssh user@login-ice.pace.gatech.edu   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ 2. Transfer code (git clone or scp)     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ 3. Submit data generation                │
│    sbatch generate_data_job.sbatch       │
│    (Wait 2-3 hours)                      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ 4. Verify data: ls training_data.json   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ 5. Submit training                       │
│    sbatch train_job.sbatch               │
│    (Wait 4-6 hours)                      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ 6. Monitor: tail -f logs/train-*.out    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ 7. Download model: scp best_*.pth       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│ 8. Test locally with trained model      │
└──────────────────────────────────────────┘
```

Good luck with your training!
