#!/bin/bash
#SBATCH --job-name=AgentPro_Train_v2
#SBATCH --partition=coc-gpu             # Use COC GPU partition (adjust if needed)
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # 1 GPU (any available)
#SBATCH --cpus-per-task=8               # CPU cores for data loading
#SBATCH --mem=32GB                      # Memory for model and data
#SBATCH --time=8:00:00                  # 8 hours (sufficient for 30k positions)
#SBATCH -o logs/train_v2-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Model Training v2 on PACE GPU
# Trains 320-hidden, 4-block residual network on 30k positions
# Improved anti-repetition heuristics in training data
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules
module load python/3.10
module load cuda/12.1

# Create or activate virtual environment
if [ ! -d "training_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv training_env
    source training_env/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install numpy pyyaml tensorboard
else
    echo "Using existing virtual environment..."
    source training_env/bin/activate
fi

# Verify GPU availability
echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo ""

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Check if training data exists
if [ ! -f "training_data_v2.json" ]; then
    echo "ERROR: training_data_v2.json not found!"
    echo "Please run generate_data_v2_job.sbatch first."
    exit 1
fi

echo "Training data found:"
ls -lh training_data_v2.json
echo ""

echo "Starting v2 model training..."
echo "Configuration: config_v2.yaml"
echo "  - Model: 320 hidden, 4 residual blocks"
echo "  - Data: 30k positions (depth 9 labels)"
echo "  - Training: 400 epochs, batch 512, mixed precision"
echo "  - Early stopping: patience 40"
echo "  - Improvements: Anti-repetition penalties"
echo ""

# Run training
python -u train_on_gpu.py --config config_v2.yaml

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Check training results
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"

    # Show final model
    if [ -f "best_evaluator.pth" ]; then
        echo ""
        echo "Best model saved:"
        ls -lh best_evaluator.pth

        # Create backup with timestamp
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        cp best_evaluator.pth "best_evaluator_v2_${TIMESTAMP}.pth"
        echo "✓ Backup created: best_evaluator_v2_${TIMESTAMP}.pth"
    fi

    # Show tensorboard logs
    if [ -d "runs_v2" ]; then
        echo ""
        echo "TensorBoard logs available in: runs_v2/"
        echo "View with: tensorboard --logdir=runs_v2"
    fi

    # Show checkpoints
    if ls checkpoint_*.pth 1> /dev/null 2>&1; then
        echo ""
        echo "Checkpoints saved:"
        ls -lh checkpoint_*.pth | tail -5
    fi
else
    echo "✗ ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check logs/train_v2-${SLURM_JOB_ID}.out for details"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Next steps:"
echo "1. Download best_evaluator.pth to your agent directory"
echo "2. Test the agent locally or in tournament"
echo "3. Review training curves: tensorboard --logdir=runs_v2"
echo "=========================================="
