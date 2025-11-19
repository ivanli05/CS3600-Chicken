# Local Testing Instructions

Test the training pipeline on your laptop before submitting to PACE.

---

## Prerequisites

1. **Python 3.10+** (required for `match` statement in game code)
2. **Dependencies**: Install from requirements.txt

---

## Step 1: Check Python Version

```bash
python --version
# Should be 3.10 or higher
```

If you need Python 3.10:
- **macOS**: `brew install python@3.10`
- **Linux**: `sudo apt install python3.10`
- **Windows**: Download from python.org

---

## Step 2: Install Dependencies

```bash
cd /Users/ivanli/Documents/GitHub/CS3600-Chicken/dist/3600-agents/AgentPro/training

# Install base requirements
pip install -r ../../../requirements.txt

# Install training dependencies
pip install pyyaml numpy torch
```

**Note**: PyTorch installation may take a while. For CPU-only (recommended for laptop):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 3: Generate Test Data (Small Dataset)

```bash
python generate_data_local.py
```

**What to expect:**
- Generates 100 positions (instead of 50,000)
- Uses 2 workers (instead of 32)
- Search depth 4 (instead of 6)
- Takes ~2-5 minutes on a modern laptop

**Output:**
```
============================================================
Local Testing - Data Generation
============================================================
Target positions: 100
Search depth: 4
Parallel workers: 2
Move range: 5-20
Output file: training_data_local.json
============================================================

Starting generation...

Progress: 10/100 (10.0%) | Rate: 3.2 pos/s | ETA: 28s
Progress: 20/100 (20.0%) | Rate: 3.5 pos/s | ETA: 23s
...
Progress: 100/100 (100.0%) | Rate: 3.8 pos/s | ETA: 0s

============================================================
Generation complete!
============================================================
Generated: 100 positions
Time: 26.3 seconds
Rate: 3.8 positions/second
============================================================

Dataset Statistics:
  Score mean: 0.042
  Score std: 0.234
  Score range: [-0.654, 0.721]

Saving to training_data_local.json...
✓ Dataset saved!
```

**Troubleshooting:**

If you see import errors, make sure you're running from the correct directory:
```bash
pwd
# Should end with: /AgentPro/training
```

If you see "SyntaxError: invalid syntax" on `match` statement:
```bash
python --version
# Must be 3.10 or higher
```

---

## Step 4: Test Training (Optional)

```bash
python train_on_gpu.py --config config_local.yaml
```

**What to expect:**
- Trains on CPU (slow but works)
- 20 epochs
- Batch size 32
- Takes ~5-10 minutes for 100 positions

**Output:**
```
============================================================
AgentB Training on H100 GPU
============================================================

Using device: cpu
Loading data from training_data_local.json...
Loaded 100 positions

Dataset splits:
  Train: 70
  Val:   15
  Test:  15

Model parameters: 49,153

============================================================
Starting training for 20 epochs
============================================================

Epoch 1/20 (2.3s) | Train: 0.1234 | Val: 0.1456 | LR: 0.001000
  ✓ New best model! (val_loss=0.1456)
Epoch 2/20 (2.1s) | Train: 0.0987 | Val: 0.1123 | LR: 0.001000
  ✓ New best model! (val_loss=0.1123)
...
```

**Note**: Training on CPU is SLOW. This is just for testing. The real training should be done on PACE's H100 GPU.

---

## Step 5: Verify Output Files

After generation:
```bash
ls -lh training_data_local.json
# Should be ~50-100 KB

# Check the content
python -c "import json; data=json.load(open('training_data_local.json')); print(f'Positions: {len(data)}')"
# Should print: Positions: 100
```

After training (if you ran it):
```bash
ls -lh best_evaluator.pth
# Should be ~200 KB
```

---

## What You're Testing

This local test verifies:
- ✅ Import paths work correctly
- ✅ Multiprocessing workers can find modules
- ✅ Board/Chicken initialization works
- ✅ Feature extraction works
- ✅ Minimax evaluation works
- ✅ Data is saved correctly
- ✅ Training script can load data

If this works locally, it will work on PACE!

---

## Differences: Local vs PACE

| Feature | Local Test | PACE Production |
|---------|-----------|-----------------|
| Positions | 100 | 50,000 |
| Workers | 2 | 32 |
| Search depth | 4 | 6 |
| Time | ~5 min | ~3 hours |
| Device | CPU | H100 GPU |
| Batch size | 32 | 256 |
| Epochs | 20 | 200 |

---

## Next Steps

Once local testing succeeds:

1. **Commit changes to Git**:
   ```bash
   cd /Users/ivanli/Documents/GitHub/CS3600-Chicken
   git add .
   git commit -m "Add PACE training pipeline with fixes"
   git push
   ```

2. **Deploy to PACE**:
   ```bash
   ssh <your-gt-username>@login-ice.pace.gatech.edu
   cd ~/projects
   git clone https://github.com/your-username/CS3600-Chicken.git
   cd CS3600-Chicken/dist/3600-agents/AgentPro/training
   sbatch generate_data_job.sbatch
   ```

3. **Monitor on PACE**:
   ```bash
   squeue -u $USER
   tail -f logs/datagen-*.out
   ```

---

## Clean Up

Remove test files when done:
```bash
rm training_data_local.json
rm best_evaluator.pth
rm -rf runs/  # TensorBoard logs
```

---

## Troubleshooting

### Error: "No module named 'game'"

**Cause**: Running from wrong directory

**Fix**:
```bash
cd /Users/ivanli/Documents/GitHub/CS3600-Chicken/dist/3600-agents/AgentPro/training
python generate_data_local.py
```

### Error: "SyntaxError: invalid syntax" (match statement)

**Cause**: Python version < 3.10

**Fix**: Install Python 3.10+
```bash
python3.10 generate_data_local.py
```

### Error: "ModuleNotFoundError: No module named 'torch'"

**Cause**: PyTorch not installed

**Fix**:
```bash
pip install torch numpy pyyaml
```

### Warning: "⚠ No trained weights found"

**Status**: NORMAL - This is expected during data generation

The agent tries to load pre-trained weights but falls back to random initialization. This doesn't affect data generation.

### Workers producing 0 positions

**Cause**: Import errors in worker processes

**Debug**:
```bash
# Look for error messages in output
python generate_data_local.py 2>&1 | grep -i error
```

---

Good luck! 🚀
