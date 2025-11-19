#!/bin/bash
#SBATCH --job-name=AgentPro_DataGen        # Job name
#SBATCH -N1 --ntasks-per-node=1            # Number of nodes and cores per node required
#SBATCH --cpus-per-task=32                 # Use many cores for parallel generation
#SBATCH --mem=32G                          # Memory
#SBATCH --time=3:00:00                     # Duration of the job (3 hours)
#SBATCH -o logs/datagen-%j.out             # Output log
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail preferences

# ============================================================================
# AgentPro Training Data Generation (Parallel)
# ============================================================================

echo "=========================================="
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
if [ ! -d "datagen_env" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv datagen_env
    source datagen_env/bin/activate
    echo "Installing dependencies..."
    pip install -r ../../../requirements.txt
    pip install pyyaml
else
    echo "Using existing virtual environment..."
    source datagen_env/bin/activate
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
