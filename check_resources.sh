#!/bin/bash
# Script to check server computation resources

echo "=========================================="
echo "CPU Information"
echo "=========================================="
lscpu | head -20
echo ""
echo "CPU Details:"
cat /proc/cpuinfo | grep -E "model name|processor|cpu cores|siblings" | head -10
echo ""

echo "=========================================="
echo "Memory Information"
echo "=========================================="
free -h
echo ""

echo "=========================================="
echo "GPU Information (if available)"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found. No NVIDIA GPUs detected or driver not installed."
fi
echo ""

echo "=========================================="
echo "System Information"
echo "=========================================="
uname -a
echo ""
hostnamectl
echo ""

echo "=========================================="
echo "SLURM Cluster Information"
echo "=========================================="
if command -v sinfo &> /dev/null; then
    echo "Available Partitions:"
    sinfo
    echo ""
    echo "Detailed Partition Info:"
    sinfo -o "%P %a %l %D %T %N %G %b" -e
    echo ""
    echo "Node Information:"
    scontrol show node 2>/dev/null | head -50 || echo "Cannot access node details (may require admin privileges)"
else
    echo "SLURM commands not available"
fi
echo ""

echo "=========================================="
echo "Current Node Resources (if in SLURM job)"
echo "=========================================="
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Job ID: $SLURM_JOB_ID"
    echo "Node: $SLURM_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "Memory allocated: $SLURM_MEM_PER_NODE"
    echo "Partition: $SLURM_JOB_PARTITION"
    echo ""
    echo "Detailed job info:"
    scontrol show job $SLURM_JOB_ID
else
    echo "Not running in a SLURM job context"
fi

