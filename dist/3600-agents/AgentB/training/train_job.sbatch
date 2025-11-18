#!/bin/bash
#SBATCH --job-name=agentb_train             # Job name
#SBATCH -N1 --ntasks-per-node=1             # Number of nodes and cores per node required
#SBATCH --gres=gpu:H100:1                   # GPU type (H100) and number of GPUs
#SBATCH --mem-per-gpu=64GB                  # Memory per GPU
#SBATCH --time=06:00:00                     # 6-hour time limit
#SBATCH -o logs/train-%j.out                # Output log (%j = job ID)
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail preferences

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

# Load Python module
module load python/3.10

# Create or activate virtual environment
if [ ! -d "agentb_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv agentb_env
    source agentb_env/bin/activate
    echo "Installing dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pyyaml tensorboard
else
    echo "Using existing virtual environment..."
    source agentb_env/bin/activate
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

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
