#!/bin/bash
#SBATCH --job-name=agentb_train      # Job name
#SBATCH --output=logs/train_%j.out   # Output log (%j = job ID)
#SBATCH --error=logs/train_%j.err    # Error log
#SBATCH --time=06:00:00              # 6-hour time limit
#SBATCH --partition=gpu              # GPU partition (adjust for your cluster)
#SBATCH --gres=gpu:h100:1            # Request 1x H100 GPU
#SBATCH --cpus-per-task=16           # CPU cores for data loading
#SBATCH --mem=64G                    # Memory
#SBATCH --ntasks=1                   # Single task

# ============================================================================
# AgentB Training on H100 GPU
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU information
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Load modules (adjust for your cluster)
module purge
module load cuda/12.1
module load cudnn/8.9
module load python/3.11

# Activate virtual environment (if using one)
# source ~/venvs/agentb/bin/activate

# Or use conda
# conda activate agentb

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to training directory
cd $SLURM_SUBMIT_DIR

echo "Working directory: $(pwd)"
echo ""

# Check if training data exists
if [ ! -f "training_data.json" ]; then
    echo "ERROR: training_data.json not found!"
    echo "Please run generate_data_job.sbatch first"
    exit 1
fi

echo "Starting training..."
echo ""

# Run training with time limit (5.5 hours to be safe)
python train_on_gpu.py \
    --config config.yaml \
    --max-hours 5.5

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Copy best model to timestamped backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -f "best_evaluator.pth" ]; then
    cp best_evaluator.pth "best_evaluator_${TIMESTAMP}.pth"
    echo "Saved backup: best_evaluator_${TIMESTAMP}.pth"
fi
