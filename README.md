# RL Elastica Snake

Reinforcement Learning training for a continuum snake using PyElastica and Stable-Baselines3.

## Project Structure

```
RL-ElasticaSnake/
├── snake_env.py          # Environment classes (BaseContinuumSnakeEnv, FixedWavelengthContinuumSnakeEnv, etc.)
├── callbacks.py          # Training callbacks (RewardCallback)
├── config.py             # Configuration parameters
├── train.py              # Main training script
├── test.py               # Model evaluation script
├── Utilities/
│   └── visualization.py  # Visualization utilities
├── slurm_job.sh          # Example Slurm job script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Training

Run training locally:
```bash
python train.py
```

### Training on Slurm Cluster

1. Modify `slurm_job.sh` with your cluster's specifications (time limits, memory, GPU requirements, etc.)

2. Submit the job:
```bash
sbatch slurm_job.sh
```

3. Monitor the job:
```bash
squeue -u $USER
```

4. Check logs:
```bash
tail -f logs/slurm_<job_id>.out
```

### Testing Trained Models

Test a trained model:
```bash
python test.py --model_path Training/Saved_Models/PPO_Snake_Model --num_steps 500
```

With stochastic actions:
```bash
python test.py --model_path Training/Saved_Models/PPO_Snake_Model --num_steps 500 --stochastic
```

## Configuration

Edit `config.py` to modify:
- Environment parameters (wavelength, observation keys, period, etc.)
- Training parameters (total timesteps, print frequency, etc.)
- Model parameters (policy type, verbosity, etc.)
- File paths (log directory, model directory, etc.)

## Environment Details

- **Action Space**: 6D continuous (torque coefficients) for fixed wavelength, 7D for variable wavelength
- **Observation Space**: Configurable (position, velocity, curvature, director, etc.)
- **Reward Function**: Combines forward progress, lateral penalty, curvature penalty, energy penalty, smoothness penalty, and alignment bonus

## Outputs

- **Models**: Saved to `Training/Saved_Models/`
- **Logs**: Training logs and Slurm output in `logs/`
- **Tensorboard**: Uncomment tensorboard_log in train.py to enable (optional)

## Notes

- The environment uses PyElastica for physics simulation
- Training can be interrupted (Ctrl+C) and the model will be saved
- Adjust `total_timesteps` in `config.py` for longer training runs
- For production training, set `total_timesteps` to 50,000 or more

