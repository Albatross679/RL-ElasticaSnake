#!/bin/bash
#SBATCH --job-name=test_python
#SBATCH --output=logs/test_python_%j.out
#SBATCH --error=logs/test_python_%j.err
#SBATCH --time=00:05:00          # 5 minutes max
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

# If needed, load modules here (many ECE nodes already have python3)
# module load python

# Create logs directory if it doesn't exist
mkdir -p logs

# Go to the folder where the script lives (use SLURM_SUBMIT_DIR or absolute path)
cd ~

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the Python script
python3 train.py