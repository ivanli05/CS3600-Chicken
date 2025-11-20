#!/bin/bash
#SBATCH --job-name=AgentPro_TimingTest
#SBATCH --partition=coc-cpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16GB
#SBATCH --time=0:30:00                  # 30 minutes is plenty
#SBATCH -o logs/timing_test-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Quick PACE Timing Test - Get REAL numbers before committing hours
# ============================================================================

echo "=========================================="
echo "PACE TIMING TEST"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

echo "Running benchmark on PACE CPUs..."
echo "This will test depths 6, 7, and 8 with 50 samples each"
echo "Expected time: 3-5 minutes"
echo ""

# Run benchmark
python -u benchmark_generation.py <<EOF

EOF

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
echo ""
echo "Check the output above for PACE-specific timings."
echo "Use those numbers to decide your final configuration."
echo ""
