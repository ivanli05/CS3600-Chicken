#!/bin/bash
#SBATCH --job-name=AgentPro_DataGen_v2
#SBATCH --partition=coc-cpu             # Use COC CPU partition
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=32              # Use all CPU cores for parallel generation
#SBATCH --mem=64GB                       # Memory for parallel workers
#SBATCH --time=8:00:00                  # 8 hours (30k @ depth 9, 0.74 pos/s = ~11 hrs, safe buffer)
#SBATCH -o logs/datagen_v2-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Training Data Generation v2 on PACE
# Generates 30k ultra-high-quality positions with depth 9 minimax labels
# Improved anti-repetition heuristics to prevent looping behavior
# Measured rate: 0.74 pos/s → 30k positions in ~11 hours
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

echo "Starting v2 data generation..."
echo "Target: 30,000 positions"
echo "Depth: 9 (ultra-high quality evaluation)"
echo "Workers: 32 (parallel)"
echo "Expected time: ~11 hours (measured: 0.74 pos/s)"
echo "Improvements: Stronger anti-repetition penalties"
echo ""

# Run data generation
python -u generate_data_v2.py

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Check if output file was created
if [ -f "training_data_v2.json" ]; then
    echo "✓ Data generation successful!"
    ls -lh training_data_v2.json

    # Create backup with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    cp training_data_v2.json "training_data_v2_${TIMESTAMP}.json"
    echo "✓ Backup created: training_data_v2_${TIMESTAMP}.json"
else
    echo "✗ ERROR: training_data_v2.json not found!"
    exit 1
fi

echo ""
echo "Next step: sbatch train_v2_job.sbatch"
