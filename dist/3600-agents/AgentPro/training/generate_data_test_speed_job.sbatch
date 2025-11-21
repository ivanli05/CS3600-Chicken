#!/bin/bash
#SBATCH --job-name=AgentPro_SpeedTest
#SBATCH --partition=coc-cpu             # Use COC CPU partition
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=32              # Use all CPU cores for parallel generation
#SBATCH --mem=64GB                       # Memory for parallel workers
#SBATCH --time=0:10:00                  # 10 minutes for quick test
#SBATCH -o logs/speedtest-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# SPEED TEST - Benchmark depth 9 performance on PACE
# Generates 100 positions to measure actual positions/second
# ============================================================================

echo "=========================================="
echo "DEPTH 9 SPEED TEST"
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

echo "Testing depth 9 generation speed..."
echo "Target: 100 positions (for accurate measurement)"
echo "Depth: 9 (ultra-high quality)"
echo "Workers: 32 (parallel)"
echo "Expected time: 1-2 minutes"
echo ""

# Run speed test
python -u generate_data_test_speed.py

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Display final rate for easy calculation
echo ""
echo "Use the final rate to calculate:"
echo "  - If rate is 2.0 pos/s → 60k positions = 8.3 hours"
echo "  - If rate is 1.5 pos/s → 60k positions = 11.1 hours"
echo "  - If rate is 1.0 pos/s → 60k positions = 16.7 hours"
echo ""
