# Updated for Your H100 Server

The sbatch files have been updated to match your server's format.

## Key Changes Made

### 1. SLURM Directives
```bash
# OLD (generic)
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --output=logs/train_%j.out

# NEW (your server)
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1           # Capital H
#SBATCH --mem-per-gpu=64GB          # mem-per-gpu format
#SBATCH -o logs/train-%j.out        # Shorter format
#SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications
```

### 2. Virtual Environment Management
```bash
# Now creates/activates venv in the script itself
if [ ! -d "agentb_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv agentb_env
    source agentb_env/bin/activate
    pip install torch ...
else
    echo "Using existing virtual environment..."
    source agentb_env/bin/activate
fi
```

### 3. Python Module
```bash
module load python/3.10  # Matches your server
```

### 4. Unbuffered Python Output
```bash
python -u train_on_gpu.py  # -u flag for real-time output
```

## Quick Start (Updated)

```bash
# 1. SSH to your server
ssh user@your-server.edu

# 2. Upload training files
cd /path/to/AgentB/training

# 3. Make sure all Python files are uploaded
# - config.yaml
# - train_on_gpu.py
# - generate_data_parallel.py
# - train_job.sbatch
# - generate_data_job.sbatch
# - monitor_training.py

# 4. Create logs directory
mkdir -p logs

# 5. Submit data generation (3 hours)
sbatch generate_data_job.sbatch

# 6. Check status
squeue -u $USER

# 7. Monitor data generation
tail -f logs/datagen-<jobid>.out

# 8. After data generation completes, submit training (6 hours)
sbatch train_job.sbatch

# 9. Monitor training (from another terminal)
python monitor_training.py
```

## What the Scripts Do Now

### `generate_data_job.sbatch`:
1. Creates virtual environment (first time only)
2. Installs numpy, pyyaml
3. Runs data generation with 32 CPU cores
4. Creates `training_data.json` (~2 hours)

### `train_job.sbatch`:
1. Creates virtual environment (first time only)
2. Installs PyTorch with CUDA 12.1, numpy, pyyaml, tensorboard
3. Checks for training_data.json
4. Trains for 5.5 hours (stops before 6h limit)
5. Saves `best_evaluator.pth`

## Email Notifications

You'll receive emails when:
- Job starts
- Job completes successfully
- Job fails

Make sure your email is configured with SLURM:
```bash
# Check current email
scontrol show job <jobid> | grep Mail

# Update email (if needed)
# Add to your ~/.bashrc:
export SBATCH_MAIL_USER=your.email@university.edu
```

## Checking Job Status

```bash
# See all your jobs
squeue -u $USER

# See specific job
squeue -j <jobid>

# Cancel a job
scancel <jobid>

# See job details
scontrol show job <jobid>

# See job history
sacct -j <jobid> --format=JobID,JobName,Partition,State,Elapsed,NodeList
```

## Log Files

```bash
# Data generation
logs/datagen-<jobid>.out

# Training
logs/train-<jobid>.out

# Real-time monitoring
tail -f logs/train-<jobid>.out
```

## File Structure After Training

```
training/
├── agentb_env/                 # Virtual environment (created by scripts)
├── logs/                       # Job logs
│   ├── datagen-12345.out
│   └── train-12346.out
├── runs/                       # TensorBoard logs
│   └── 20250118_143022/
├── training_data.json          # Generated data (~500MB)
├── best_evaluator.pth          # Best model
├── best_evaluator_20250118_180530.pth  # Timestamped backup
├── latest_checkpoint.pth       # Latest checkpoint (resume)
├── checkpoint_epoch_100.pth    # Periodic checkpoints
└── training_progress.png       # Loss curves (from monitor)
```

## Troubleshooting

### "Module python/3.10 not found"
```bash
# Check available Python modules
module avail python

# Update sbatch files to use available version
# e.g., if you have python/3.11:
module load python/3.11
python3.11 -m venv agentb_env
```

### "GPU not found"
```bash
# Check GPU availability
sinfo -o "%20N %10c %10m %25f %10G"

# Update sbatch file if using different GPU
#SBATCH --gres=gpu:A100:1  # If using A100 instead
```

### "Out of memory"
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 128  # Or 64
```

### Resume from Checkpoint
If training was interrupted:
```bash
# Edit train_job.sbatch, change last command to:
python -u train_on_gpu.py \
    --config config.yaml \
    --resume latest_checkpoint.pth \
    --max-hours 5.5
```

## Performance Expectations

With your H100 GPU:
- **Data generation**: 2-2.5 hours (32 CPUs)
- **Training**: 3.5-4 hours (200 epochs)
- **Total**: ~6 hours

## Next Steps After Training

```bash
# 1. Download trained model
scp user@server:/path/to/training/best_evaluator.pth ./

# 2. Copy to agent directory
cp best_evaluator.pth ../evaluator_weights.pth

# 3. Test locally
cd ../../../
python engine/run_local_agents.py AgentB AgentB

# Look for this line:
# ✓ Loaded trained evaluator weights from evaluator_weights.pth
```

## Tips

1. **First run**: Scripts will create virtual environment and install packages (takes 5-10 min extra)
2. **Subsequent runs**: Uses existing venv (no extra time)
3. **Monitor in real-time**: `python monitor_training.py` from login node
4. **TensorBoard**: If your cluster supports it:
   ```bash
   tensorboard --logdir=runs --port=6006 --bind_all
   ```

Good luck with your training! 🚀
