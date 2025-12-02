#!/bin/bash
# Quick script to check PACE partition availability

echo "======================================"
echo "Checking PACE partition availability"
echo "======================================"
echo ""

echo "Available partitions:"
sinfo -s
echo ""

echo "Detailed partition info:"
sinfo -l
echo ""

echo "Your current job status:"
squeue -u $USER
echo ""

echo "======================================"
echo "Suggested actions:"
echo "======================================"
echo ""
echo "1. Wait for coc-cpu partition to become available"
echo "2. Try alternate partition (see options below)"
echo "3. Contact PACE support if nodes stay DOWN"
echo ""
echo "Common CPU partitions at PACE:"
echo "  - coc-cpu      (COC cluster CPUs)"
echo "  - cpu-medium   (General CPU, up to 24hrs)"
echo "  - cpu-small    (General CPU, up to 2hrs)"
echo ""
