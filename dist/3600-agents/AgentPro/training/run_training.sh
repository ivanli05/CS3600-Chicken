#!/bin/bash
# Quick training script for AgentB

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "AgentB Training Helper"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Check if training data exists
if [ ! -f "training_data.json" ]; then
    echo -e "${YELLOW}No training data found!${NC}"
    echo ""
    echo "Options:"
    echo "  1. Generate data with SLURM: sbatch generate_data_job.sbatch"
    echo "  2. Generate data locally:    python generate_data_parallel.py"
    echo ""
    read -p "Generate data now? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Submitting data generation job..."
        sbatch generate_data_job.sbatch
        echo ""
        echo -e "${GREEN}Job submitted!${NC}"
        echo "Monitor with: squeue -u \$USER"
        echo "Or check: tail -f logs/datagen_*.out"
        exit 0
    else
        echo "Please generate data first, then run this script again."
        exit 1
    fi
fi

# Data exists, proceed with training
echo -e "${GREEN}Training data found!${NC}"
DATA_SIZE=$(du -h training_data.json | cut -f1)
echo "  File: training_data.json ($DATA_SIZE)"
echo ""

# Check if already training
RUNNING_JOBS=$(squeue -u $USER -n agentb_train -h | wc -l)
if [ $RUNNING_JOBS -gt 0 ]; then
    echo -e "${YELLOW}Training job already running!${NC}"
    echo ""
    squeue -u $USER -n agentb_train
    echo ""
    read -p "Monitor training? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python monitor_training.py
    fi

    exit 0
fi

# Submit training job
echo "Submitting training job..."
JOB_ID=$(sbatch train_job.sbatch | awk '{print $4}')
echo ""
echo -e "${GREEN}Training job submitted!${NC}"
echo "  Job ID: $JOB_ID"
echo ""
echo "Commands:"
echo "  Monitor: python monitor_training.py"
echo "  Status:  squeue -j $JOB_ID"
echo "  Cancel:  scancel $JOB_ID"
echo "  Logs:    tail -f logs/train_${JOB_ID}.out"
echo ""

read -p "Start monitoring now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Wait a bit for log file to be created
    echo "Waiting for training to start..."
    sleep 10

    python monitor_training.py --log-file logs/train_${JOB_ID}.out
fi
