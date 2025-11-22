# RL Elastica Snake: Project Overview

## Project Overview

This project implements **reinforcement learning (RL) control for a continuum snake robot** using physics-based simulation. The goal is to train an agent that learns optimal locomotion strategies for a flexible, continuum snake-like robot navigating on a surface with friction.

### Key Highlights
• **RL-driven locomotion**: Uses PPO (Proximal Policy Optimization) to learn optimal control policies for a continuum snake robot navigating with friction
• **Physics-accurate simulation**: Leverages PyElastica for realistic continuum rod dynamics, including contact forces, muscle torques, and anisotropic friction
• **Multi-objective learning**: Trains the agent to balance forward progress, directional alignment, energy efficiency, and smooth motion through a composite reward function

## Core Objective

Train a Proximal Policy Optimization (PPO) agent to control a continuum snake robot to achieve:
- **Efficient forward locomotion** along a target direction
- **Minimal lateral drift** and energy consumption
- **Stable, smooth motion** with controlled curvature
- **Directional alignment** with a target heading

## Technical Architecture

### Physics Simulation
- **Framework**: PyElastica (v0.3.3) for continuum rod dynamics
- **Robot Model**: Cosserat rod with 50 elements
- **Environment**: Ground plane with anisotropic friction (Froude number = 0.1)
- **Forces**: Gravity, muscle torques (periodic actuation), damping, contact forces

### Reinforcement Learning
- **Algorithm**: PPO (Proximal Policy Optimization) from Stable-Baselines3
- **Action Space**: 6D continuous (torque coefficients) for fixed-wavelength control
- **Observation Space**: Configurable multi-dimensional state including:
  - Position and velocity (node-level and center-of-mass)
  - Curvature and tangents
  - Director matrices (orientation)
  - Time information

### Reward Design
Multi-component reward function balancing:
- **Forward Progress**: Reward for movement in target direction
- **Lateral Penalty**: Discourage sideways drift
- **Curvature Penalty**: Encourage smooth, controlled motion
- **Energy Penalty**: Minimize torque magnitudes (currently disabled)
- **Smoothness Penalty**: Penalize abrupt torque changes between steps
- **Alignment Bonus**: Reward maintaining target heading direction
- **Streak Bonus**: Large bonus for sustained alignment (50+ consecutive steps)
- **Projected Speed**: Reward velocity component in target direction

## Key Features

### Environment Variants
- **FixedWavelengthContinuumSnakeEnv**: Wavelength fixed at 1.0 (6D action space)
- **VariableWavelengthContinuumSnakeEnv**: Wavelength learned as 7th action dimension

### Training Infrastructure
- **Checkpoint System**: Periodic model saves (default: every 10k timesteps)
- **Resume Training**: Ability to continue from saved checkpoints
- **Logging**: Comprehensive episode and step-level metrics
- **SLURM Support**: Cluster job submission scripts included
- **Signal Handling**: Graceful shutdown on SIGTERM/SIGINT with model saving

### Evaluation & Analysis
- **Model Testing**: Deterministic and stochastic evaluation modes
- **Logging Metrics**: Forward speed, lateral speed, velocity projection, alignment streak
- **Visualization**: Training log analysis and plotting tools

## Training Workflow

1. **Configuration**: Edit `config.py` for environment, training, and model parameters
2. **Training**: Run `train.py` for new training or `resume_train.py` to continue
3. **Evaluation**: Use `test.py` to evaluate trained models
4. **Analysis**: Visualize training logs using provided notebooks

## Current Configuration

- **Fixed Wavelength**: 1.0
- **Period**: 1.0
- **Observations**: Position, velocity, curvature
- **Target Training**: 5,000 timesteps (configurable to 50k+ for production)
- **Episode Termination**: Max simulation time (300s) or successful alignment streak (50 steps)

## Research Context

This work explores learning-based control for **continuum robotics**, specifically snake-like locomotion. The physics-accurate simulation enables transfer to real-world applications in:
- Search and rescue robotics
- Medical snake robots
- Inspection in confined spaces
- Bio-inspired locomotion research

