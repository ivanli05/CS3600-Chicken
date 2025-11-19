#!/bin/bash
#SBATCH --job-name=AgentPro_Training       # Job name
#SBATCH -N1 --ntasks-per-node=1            # Number of nodes and cores per node required
#SBATCH --gres=gpu:H100:1                  # GPU type (H100) and number of GPUs
#SBATCH --mem-per-gpu=64GB                 # Memory per GPU
#SBATCH --cpus-per-task=16                 # CPU cores for data loading
#SBATCH --time=6:00:00                     # Duration of the job (6 hours)
#SBATCH -o logs/train-%j.out               # Output log
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail preferences

# ============================================================================
# AgentPro Training on H100 GPU
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

# Load modules
module load python/3.10

# Create or activate virtual environment
if [ ! -d "training_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv training_env
    source training_env/bin/activate
    echo "Installing dependencies..."
    pip install -r ../../../requirements.txt
    pip install pyyaml tensorboard matplotlib
else
    echo "Using existing virtual environment..."
    source training_env/bin/activate
fi

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
python -u train_on_gpu.py \
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
