# AgentB Training on H100 GPU

Complete guide for training AgentB's neural network evaluator on H100 GPU via SLURM.

---

## Quick Start

```bash
# 1. SSH into your H100 server
ssh user@your-server.edu

# 2. Navigate to training directory
cd /path/to/AgentB/training

# 3. Create logs directory
mkdir -p logs

# 4. Generate training data (2-3 hours)
sbatch generate_data_job.sbatch

# 5. Wait for data generation to complete, then train (4 hours)
sbatch train_job.sbatch

# 6. Monitor progress (from login node)
python monitor_training.py

# 7. Download trained model
scp user@server:/path/to/AgentB/training/best_evaluator.pth ./
```

---

## Files Overview

### Configuration
- **`config.yaml`** - All hyperparameters in one place
  - Batch size: 256 (optimized for H100)
  - Epochs: 200 (~4 hours)
  - Learning rate: 0.001 with ReduceLROnPlateau
  - Mixed precision: Enabled (FP16 for 2x speed)

### Training Scripts
- **`train_on_gpu.py`** - Main training script
  - H100-optimized with mixed precision
  - Automatic checkpointing every 30 minutes
  - TensorBoard logging
  - Time-limited (stops 30min before SLURM kills it)

- **`generate_data_parallel.py`** - Parallel data generation
  - Uses all CPU cores
  - Depth-6 search for ground truth labels
  - Generates 50k positions in ~2 hours

### SLURM Scripts
- **`train_job.sbatch`** - Training job
  - 6-hour time limit
  - 1x H100 GPU
  - 16 CPU cores for data loading
  - 64GB RAM

- **`generate_data_job.sbatch`** - Data generation job
  - 3-hour time limit
  - CPU-only (32 cores)
  - No GPU needed

### Monitoring
- **`monitor_training.py`** - Real-time progress monitor
  - Plots loss curves
  - Shows current metrics
  - Estimates completion time

---

## Detailed Workflow

### Step 1: Setup Environment

```bash
# Create virtual environment (if not exists)
python3 -m venv ~/venvs/agentb
source ~/venvs/agentb/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pyyaml tensorboard matplotlib

# Or use conda
conda create -n agentb python=3.11
conda activate agentb
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy pyyaml tensorboard matplotlib
```

### Step 2: Configure for Your Cluster

Edit `train_job.sbatch` and `generate_data_job.sbatch`:

```bash
# Change partition names to match your cluster
#SBATCH --partition=gpu          # Change to your GPU partition name
#SBATCH --gres=gpu:h100:1        # Or gpu:a100:1, etc.

# Adjust module loading
module load cuda/12.1            # Your CUDA version
module load python/3.11          # Your Python version

# Update paths to your virtual environment
source ~/venvs/agentb/bin/activate
```

### Step 3: Generate Training Data

**Option A: Submit as SLURM job (recommended)**
```bash
sbatch generate_data_job.sbatch

# Check status
squeue -u $USER

# Monitor output
tail -f logs/datagen_<jobid>.out
```

**Option B: Run interactively (if you have an interactive node)**
```bash
# Request interactive session
srun --partition=cpu --cpus-per-task=32 --mem=32G --time=3:00:00 --pty bash

# Run generation
python generate_data_parallel.py --config config.yaml
```

**Expected output:**
```
========================================
Parallel Data Generation
========================================
Target positions: 50000
Search depth: 6
Parallel workers: 32
========================================

Progress: 5000/50000 (10.0%) | Rate: 12.3 pos/s | ETA: 60.7 min
Progress: 10000/50000 (20.0%) | Rate: 11.8 pos/s | ETA: 56.4 min
...
Progress: 50000/50000 (100.0%) | Rate: 12.1 pos/s | ETA: 0.0 min

Generation complete!
Generated: 50000 positions
Time: 68.7 minutes
```

### Step 4: Train Model

```bash
# Submit training job
sbatch train_job.sbatch

# Check status
squeue -u $USER

# Check GPU usage
srun --jobid=<jobid> nvidia-smi
```

### Step 5: Monitor Progress

**From login node:**
```bash
# Live monitoring (updates every 30 seconds)
python monitor_training.py

# Single snapshot
python monitor_training.py --once

# Specify log file
python monitor_training.py --log-file logs/train_12345.out
```

**Via TensorBoard (if accessible):**
```bash
# On server
tensorboard --logdir=runs --port=6006

# On your local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@server.edu

# Open in browser
http://localhost:6006
```

**Example output:**
```
========================================
Training Summary
========================================
Current Epoch: 120/200
Progress: 60.0%

Latest Metrics:
  Train Loss: 0.0342
  Val Loss:   0.0389
  LR:         0.000500

Best Val Loss: 0.0378 (Epoch 115)

Improvement: +12.3% (last 10 epochs vs first 10)
========================================
```

### Step 6: Download Results

```bash
# Download best model
scp user@server:/path/to/AgentB/training/best_evaluator.pth ./

# Download logs
scp user@server:/path/to/AgentB/training/logs/train_*.out ./

# Download plots
scp user@server:/path/to/AgentB/training/training_progress.png ./

# Or download entire training directory
scp -r user@server:/path/to/AgentB/training/runs ./tensorboard_logs
```

### Step 7: Use Trained Model

```bash
# Copy to agent directory
cp best_evaluator.pth ../evaluator_weights.pth

# Test agent
cd ../../../
python engine/run_local_agents.py AgentB AgentB

# Look for this in output:
# ✓ Loaded trained evaluator weights from evaluator_weights.pth
```

---

## Configuration Tuning

### For Shorter Training (~2 hours)
```yaml
# Edit config.yaml
training:
  epochs: 100  # Reduce from 200

data:
  num_positions: 25000  # Reduce from 50000
```

### For More Data (~8-10 hours total)
```yaml
data_generation:
  num_positions: 100000  # 100k positions

training:
  epochs: 300
```

### For Faster Training (Lower Quality)
```yaml
training:
  batch_size: 512  # Increase batch size

data_generation:
  depth_for_labels: 4  # Reduce search depth (faster, less accurate)
```

### For Better Quality (Slower)
```yaml
data_generation:
  depth_for_labels: 8  # Deeper search (slower, more accurate)

training:
  batch_size: 128  # Smaller batch (better gradients)
```

---

## Troubleshooting

### Job Fails Immediately
```bash
# Check output log
cat logs/train_<jobid>.err

# Common issues:
# 1. Wrong partition name → Edit sbatch file
# 2. Module not found → Check available modules: module avail
# 3. CUDA version mismatch → Match CUDA with PyTorch version
```

### Out of Memory (OOM)
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 128  # Or 64
```

### Training Too Slow
```bash
# Check GPU utilization
nvidia-smi

# If GPU usage < 80%:
# - Increase batch_size
# - Increase num_workers (data loading)
# - Check if CPU-bound (data generation bottleneck)
```

### Data Generation Too Slow
```bash
# Increase parallel workers
python generate_data_parallel.py --num-workers 64

# Or reduce search depth (less accurate)
# Edit config.yaml: depth_for_labels: 4
```

### Training Doesn't Improve
```
# Check learning rate
# If LR too high → Loss oscillates
# If LR too low → Loss doesn't decrease

# Try different learning rate
training:
  learning_rate: 0.0005  # Reduce from 0.001
```

### Resume from Checkpoint
```bash
# If job was killed/interrupted
python train_on_gpu.py \
    --config config.yaml \
    --resume latest_checkpoint.pth \
    --max-hours 5.5
```

---

## Performance Benchmarks

### H100 GPU (Expected)
- **Training speed**: ~50-60 epochs/hour
- **Total training time**: 3-4 hours for 200 epochs
- **Data generation**: ~2 hours for 50k positions (32 CPU cores)
- **Inference**: ~0.001s per position (1000x faster than depth-6 search)

### A100 GPU (Expected)
- **Training speed**: ~35-45 epochs/hour
- **Total training time**: 4-5 hours for 200 epochs

### V100 GPU (Expected)
- **Training speed**: ~25-30 epochs/hour
- **Total training time**: 6-8 hours for 200 epochs

---

## Advanced: Multi-GPU Training

If you have access to multiple GPUs:

```python
# Edit train_on_gpu.py
# Add DataParallel wrapper
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

```bash
# Edit train_job.sbatch
#SBATCH --gres=gpu:h100:4  # Request 4 GPUs
```

---

## Expected Results

### After 6 Hours of Training:

**Training Metrics:**
```
Epoch 200/200
Train Loss: 0.025-0.035
Val Loss:   0.030-0.040
Test Loss:  0.032-0.042
```

**Performance:**
- Correlation with depth-6 search: >0.85
- Mean absolute error: <0.08
- Inference speed: ~1000 evals/second

**In-Game Performance:**
- Should play better than heuristic-only agent
- Comparable to depth-6 search at depth-4 speed
- Win rate vs random: >90%
- Win rate vs heuristic-only: 60-70%

---

## Next Steps

1. **Test trained model** against heuristic-only version
2. **Generate more data** if performance isn't good enough
3. **Try reinforcement learning** for further improvement
4. **Tune hyperparameters** based on initial results
5. **Submit to tournament!**

---

## Support

If you encounter issues:
1. Check SLURM output logs: `logs/train_*.err`
2. Verify GPU is allocated: `srun --jobid=<id> nvidia-smi`
3. Check Python environment: `which python`, `pip list`
4. Review config: `cat config.yaml`

Good luck with your training! 🚀
