#!/bin/bash
#SBATCH --job-name=agentb_datagen           # Job name
#SBATCH -N1 --ntasks-per-node=1             # Number of nodes and cores per node required
#SBATCH --cpus-per-task=32                  # CPU cores for parallel generation
#SBATCH --mem=32GB                          # Memory
#SBATCH --time=03:00:00                     # 3-hour time limit
#SBATCH -o logs/datagen-%j.out              # Output log (%j = job ID)
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail preferences

# ============================================================================
# AgentB Training Data Generation (Parallel)
# ============================================================================

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Load Python module
module load python/3.10

# Create or activate virtual environment
if [ ! -d "agentb_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv agentb_env
    source agentb_env/bin/activate
    echo "Installing dependencies..."
    pip install numpy pyyaml
else
    echo "Using existing virtual environment..."
    source agentb_env/bin/activate
fi

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to training directory
cd $SLURM_SUBMIT_DIR

echo "Working directory: $(pwd)"
echo ""

echo "Starting parallel data generation..."
echo ""

# Run data generation with all available cores
python -u generate_data_parallel.py \
    --config config.yaml \
    --output training_data.json \
    --num-workers $SLURM_CPUS_PER_TASK

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

# Show file size
if [ -f "training_data.json" ]; then
    SIZE=$(du -h training_data.json | cut -f1)
    echo "Generated file: training_data.json ($SIZE)"
    echo ""
    echo "Ready for training! Submit train_job.sbatch"
fi
