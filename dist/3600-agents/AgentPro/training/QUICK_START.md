# Quick Start Guide

## Local Testing (Your Laptop)

```bash
# 1. Install dependencies
pip install -r ../../../requirements.txt
pip install pyyaml numpy torch

# 2. Generate test data (100 positions, ~5 minutes)
python generate_data_local.py

# 3. Optional: Test training (slow on CPU)
python train_on_gpu.py --config config_local.yaml
```

---

## PACE Production (H100 GPU)

```bash
# 1. SSH to PACE
ssh <your-gt-username>@login-ice.pace.gatech.edu

# 2. Clone/pull code
cd ~/projects
git clone https://github.com/your-username/CS3600-Chicken.git
# OR: git pull  (if already cloned)

# 3. Navigate to training directory
cd CS3600-Chicken/dist/3600-agents/AgentPro/training

# 4. Create logs directory
mkdir -p logs

# 5. Submit data generation job (50k positions, ~3 hours)
sbatch generate_data_job.sbatch

# 6. Monitor progress
squeue -u $USER
tail -f logs/datagen-*.out

# 7. After data generation completes, submit training job (~6 hours)
sbatch train_job.sbatch

# 8. Monitor training
tail -f logs/train-*.out
```

---

## File Overview

| File | Purpose | Use Where |
|------|---------|-----------|
| `generate_data_local.py` | Generate 100 test positions | Laptop |
| `generate_data_parallel.py` | Generate 50k positions | PACE |
| `train_on_gpu.py` | Train neural network | Both |
| `config_local.yaml` | Config for laptop testing | Laptop |
| `config.yaml` | Config for PACE training | PACE |
| `generate_data_job.sbatch` | SLURM job for data gen | PACE |
| `train_job.sbatch` | SLURM job for training | PACE |

---

## Expected Output Sizes

| File | Local | PACE |
|------|-------|------|
| `training_data*.json` | ~100 KB | ~100 MB |
| `best_evaluator.pth` | ~200 KB | ~1-5 MB |
| Log files | ~50 KB | ~5 MB |

---

## Time Estimates

| Task | Local | PACE |
|------|-------|------|
| Data generation | ~5 min | ~3 hours |
| Training | ~10 min | ~6 hours |
| **Total** | **~15 min** | **~9 hours** |

---

## Detailed Instructions

- **Local testing**: See `LOCAL_TEST_INSTRUCTIONS.md`
- **PACE deployment**: See `PACE_INSTRUCTIONS.md`
- **Import debugging**: See `IMPORT_DEBUG_SUMMARY.md`
- **Training details**: See `README_GPU_TRAINING.md`
