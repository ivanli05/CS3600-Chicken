#!/bin/bash
#SBATCH --job-name=AgentPro_DataGen_Ultimate
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=32              # Use all CPU cores for parallel generation
#SBATCH --mem=64GB                       # Memory for parallel workers
#SBATCH --time=8:00:00                  # 8 hours (200k positions takes ~4-6 hours)
#SBATCH -o logs/datagen_ultimate-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# ULTIMATE Training Data Generation on PACE
# Generates 200k high-quality positions with proper labels
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load modules
module load python/3.10

# Create or activate virtual environment
if [ ! -d "training_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv training_env
    source training_env/bin/activate
    echo "Installing dependencies..."
    pip install numpy
else
    echo "Using existing virtual environment..."
    source training_env/bin/activate
fi

# Set environment for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

echo "Starting ULTIMATE data generation..."
echo "Target: 200,000 positions"
echo "Depth: 8 (deep evaluation)"
echo "Workers: 32 (parallel)"
echo ""

# Run data generation
python -u generate_data_ultimate.py

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Check if output file was created
if [ -f "training_data_ultimate.json" ]; then
    echo "✓ Data generation successful!"
    ls -lh training_data_ultimate.json

    # Create backup with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    cp training_data_ultimate.json "training_data_ultimate_${TIMESTAMP}.json"
    echo "✓ Backup created: training_data_ultimate_${TIMESTAMP}.json"
else
    echo "✗ ERROR: training_data_ultimate.json not found!"
    exit 1
fi

echo ""
echo "Next step: sbatch train_ultimate_job.sbatch"
