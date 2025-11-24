"""
Visualization utilities for policy gradients and derivatives.

This module provides functions to visualize policy gradients, Jacobians,
and sensitivity analysis for trained PPO models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, List, Tuple
from pathlib import Path
from stable_baselines3 import PPO

import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from .policy_gradient_utils import (
    compute_policy_gradient,
    compute_policy_jacobian,
    get_policy_sensitivity,
)


def plot_jacobian_heatmap(
    model: PPO,
    state: Union[np.ndarray, List[np.ndarray]],
    state_labels: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a heatmap of the policy Jacobian matrix.
    
    Args:
        model: Trained PPO model
        state: Single state or list of states to visualize
        state_labels: Optional labels for state dimensions
        action_labels: Optional labels for action dimensions
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        show: Whether to display the figure
        
    Returns:
        matplotlib Figure object
    """
    # Handle single state or list of states
    if isinstance(state, list):
        states = state
    else:
        states = [state]
    
    n_states = len(states)
    fig, axes = plt.subplots(1, n_states, figsize=(figsize[0] * n_states, figsize[1]))
    
    if n_states == 1:
        axes = [axes]
    
    for idx, s in enumerate(states):
        # Compute Jacobian
        jacobian = compute_policy_jacobian(model, s)
        action_dim, state_dim = jacobian.shape
        
        # Create heatmap
        im = axes[idx].imshow(jacobian, aspect='auto', cmap='RdBu_r', 
                             interpolation='nearest')
        
        # Set labels
        if state_labels is None:
            state_labels = [f'State {i}' for i in range(state_dim)]
        if action_labels is None:
            action_labels = [f'Action {i}' for i in range(action_dim)]
        
        axes[idx].set_xticks(np.arange(state_dim))
        axes[idx].set_yticks(np.arange(action_dim))
        axes[idx].set_xticklabels(state_labels, rotation=45, ha='right')
        axes[idx].set_yticklabels(action_labels)
        axes[idx].set_xlabel('State Dimension')
        axes[idx].set_ylabel('Action Dimension')
        axes[idx].set_title(f'Policy Jacobian Matrix\nState {idx + 1}')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[idx], label='Gradient Value')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_gradient_magnitude(
    model: PPO,
    states: Union[np.ndarray, List[np.ndarray]],
    wrt: str = 'action_mean',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the magnitude of gradients across different states.
    
    Args:
        model: Trained PPO model
        states: Array of states (n_states, state_dim) or list of states
        wrt: What to compute gradient w.r.t. ('action_mean', 'value')
        figsize: Figure size
        save_path: Optional path to save the figure
        show: Whether to display the figure
        
    Returns:
        matplotlib Figure object
    """
    # Convert to numpy array if list
    if isinstance(states, list):
        states = np.array(states)
    
    if states.ndim == 1:
        states = states[np.newaxis, :]
    
    n_states = states.shape[0]
    magnitudes = []
    
    # Compute gradients for each state
    for state in states:
        if wrt == 'action_mean':
            grad_info = compute_policy_gradient(model, state, wrt_action_mean=True)
            grad = grad_info['action_mean_grad']
        elif wrt == 'value':
            grad_info = compute_policy_gradient(model, state, wrt_value=True)
            grad = grad_info['value_grad']
        else:
            raise ValueError(f"wrt must be 'action_mean' or 'value', got {wrt}")
        
        # Compute Frobenius norm for action_mean, L2 norm for value
        if wrt == 'action_mean':
            magnitude = np.linalg.norm(grad, ord='fro')
        else:
            magnitude = np.linalg.norm(grad)
        
        magnitudes.append(magnitude)
    
    magnitudes = np.array(magnitudes)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(n_states), magnitudes, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('State Index')
    ax.set_ylabel(f'Gradient Magnitude ({wrt})')
    ax.set_title(f'Gradient Magnitude Across States\n(w.r.t. {wrt})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_sensitivity_analysis(
    model: PPO,
    state: Union[np.ndarray, List[np.ndarray]],
    perturbation_directions: Optional[List[np.ndarray]] = None,
    epsilon: float = 1e-5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize policy sensitivity to state perturbations.
    
    Args:
        model: Trained PPO model
        state: Single state or list of states
        perturbation_directions: Optional list of perturbation directions
        epsilon: Perturbation magnitude for numerical differentiation
        figsize: Figure size
        save_path: Optional path to save the figure
        show: Whether to display the figure
        
    Returns:
        matplotlib Figure object
    """
    # Handle single state or list
    if isinstance(state, list):
        states = state
    else:
        states = [state]
    
    n_states = len(states)
    fig, axes = plt.subplots(2, n_states, figsize=(figsize[0] * n_states, figsize[1] * 2))
    
    if n_states == 1:
        axes = axes[:, np.newaxis]
    
    for idx, s in enumerate(states):
        # Compute sensitivity
        if perturbation_directions is not None:
            pert = perturbation_directions[idx] if idx < len(perturbation_directions) else None
        else:
            pert = None
        
        sensitivity = get_policy_sensitivity(model, s, pert, epsilon)
        
        # Plot 1: Gradient magnitude
        grad = sensitivity['gradient']
        if grad.ndim == 2:  # Action mean gradient (action_dim, state_dim)
            grad_magnitude = np.linalg.norm(grad, axis=0)  # Magnitude per state dimension
        else:  # Value gradient (state_dim,)
            grad_magnitude = np.abs(grad)
        
        axes[0, idx].bar(range(len(grad_magnitude)), grad_magnitude)
        axes[0, idx].set_xlabel('State Dimension')
        axes[0, idx].set_ylabel('Gradient Magnitude')
        axes[0, idx].set_title(f'Gradient Magnitude per State Dim\nState {idx + 1}')
        axes[0, idx].grid(True, alpha=0.3)
        
        # Plot 2: Sensitivity norm
        sensitivity_norm = sensitivity['sensitivity_norm']
        axes[1, idx].bar([0], [sensitivity_norm], width=0.5)
        axes[1, idx].set_ylabel('Sensitivity Norm')
        axes[1, idx].set_title(f'Overall Sensitivity Norm\nState {idx + 1}')
        axes[1, idx].set_xticks([])
        axes[1, idx].grid(True, alpha=0.3, axis='y')
        
        # Add numerical sensitivity if available
        if 'action_change' in sensitivity:
            action_change = sensitivity['action_change']
            axes[1, idx].text(0, sensitivity_norm, 
                             f'Num: {np.linalg.norm(action_change):.4f}',
                             ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_action_gradient_components(
    model: PPO,
    state: Union[np.ndarray, List[np.ndarray]],
    action_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot gradient components for each action dimension.
    
    Args:
        model: Trained PPO model
        state: Single state or list of states
        action_idx: Optional specific action index to plot. If None, plots all.
        figsize: Figure size
        save_path: Optional path to save the figure
        show: Whether to display the figure
        
    Returns:
        matplotlib Figure object
    """
    # Handle single state or list
    if isinstance(state, list):
        states = state
    else:
        states = [state]
    
    n_states = len(states)
    
    # Determine which actions to plot
    grad_info = compute_policy_gradient(model, states[0], wrt_action_mean=True)
    action_dim = grad_info['action_mean_grad'].shape[0]
    
    if action_idx is not None:
        action_indices = [action_idx]
    else:
        action_indices = list(range(action_dim))
    
    n_actions = len(action_indices)
    fig, axes = plt.subplots(n_actions, n_states, 
                            figsize=(figsize[0] * n_states, figsize[1] * n_actions))
    
    if n_actions == 1:
        axes = axes[np.newaxis, :]
    if n_states == 1:
        axes = axes[:, np.newaxis]
    
    for state_idx, s in enumerate(states):
        grad_info = compute_policy_gradient(model, s, wrt_action_mean=True)
        jacobian = grad_info['action_mean_grad']  # (action_dim, state_dim)
        
        for action_idx_local, action_idx_global in enumerate(action_indices):
            grad_components = jacobian[action_idx_global, :]
            
            ax = axes[action_idx_local, state_idx]
            ax.bar(range(len(grad_components)), grad_components)
            ax.set_xlabel('State Dimension')
            ax.set_ylabel('Gradient Value')
            ax.set_title(f'Action {action_idx_global} Gradient\nState {state_idx + 1}')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_gradient_comparison(
    model: PPO,
    states: List[np.ndarray],
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare gradients across multiple states.
    
    Args:
        model: Trained PPO model
        states: List of states to compare
        labels: Optional labels for each state
        figsize: Figure size
        save_path: Optional path to save the figure
        show: Whether to display the figure
        
    Returns:
        matplotlib Figure object
    """
    n_states = len(states)
    if labels is None:
        labels = [f'State {i+1}' for i in range(n_states)]
    
    # Compute gradients for all states
    jacobians = []
    for state in states:
        jacobian = compute_policy_jacobian(model, state)
        jacobians.append(jacobian)
    
    action_dim, state_dim = jacobians[0].shape
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Frobenius norm comparison
    norms = [np.linalg.norm(j, ord='fro') for j in jacobians]
    axes[0].bar(range(n_states), norms, alpha=0.7)
    axes[0].set_xticks(range(n_states))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_ylabel('Frobenius Norm')
    axes[0].set_title('Jacobian Matrix Norm Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Element-wise comparison (average absolute gradient)
    avg_abs_grads = [np.mean(np.abs(j)) for j in jacobians]
    axes[1].bar(range(n_states), avg_abs_grads, alpha=0.7)
    axes[1].set_xticks(range(n_states))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Absolute Gradient')
    axes[1].set_title('Average Absolute Gradient Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_value_gradient(
    model: PPO,
    state: Union[np.ndarray, List[np.ndarray]],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the gradient of the value function with respect to state.
    
    Args:
        model: Trained PPO model
        state: Single state or list of states
        figsize: Figure size
        save_path: Optional path to save the figure
        show: Whether to display the figure
        
    Returns:
        matplotlib Figure object
    """
    # Handle single state or list
    if isinstance(state, list):
        states = state
    else:
        states = [state]
    
    n_states = len(states)
    fig, axes = plt.subplots(1, n_states, figsize=(figsize[0] * n_states, figsize[1]))
    
    if n_states == 1:
        axes = [axes]
    
    for idx, s in enumerate(states):
        grad_info = compute_policy_gradient(model, s, wrt_value=True)
        value_grad = grad_info['value_grad']
        value = grad_info['value']
        
        # Plot gradient components
        axes[idx].bar(range(len(value_grad)), value_grad)
        axes[idx].set_xlabel('State Dimension')
        axes[idx].set_ylabel('Gradient Value')
        axes[idx].set_title(f'Value Function Gradient\nState {idx + 1} (V={value[0,0]:.3f})')
        axes[idx].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_comprehensive_report(
    model: PPO,
    states: List[np.ndarray],
    state_labels: Optional[List[str]] = None,
    output_dir: Union[str, Path] = None,
    show: bool = False,
) -> None:
    """
    Create a comprehensive visualization report for policy gradients.
    
    Args:
        model: Trained PPO model
        states: List of states to analyze
        state_labels: Optional labels for states
        output_dir: Directory to save visualizations (defaults to config.PATHS["policy_gradient_viz_dir"])
        show: Whether to display plots interactively
    """
    if output_dir is None:
        output_dir = config.PATHS["policy_gradient_viz_dir"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if state_labels is None:
        state_labels = [f'State {i+1}' for i in range(len(states))]
    
    print(f"Creating policy gradient visualizations in {output_dir}...")
    
    # 1. Jacobian heatmaps
    print("  - Creating Jacobian heatmaps...")
    plot_jacobian_heatmap(
        model, states, 
        save_path=output_dir / "jacobian_heatmap.png",
        show=show
    )
    
    # 2. Gradient magnitude
    print("  - Creating gradient magnitude plot...")
    plot_gradient_magnitude(
        model, states,
        save_path=output_dir / "gradient_magnitude.png",
        show=show
    )
    
    # 3. Sensitivity analysis
    print("  - Creating sensitivity analysis...")
    plot_sensitivity_analysis(
        model, states,
        save_path=output_dir / "sensitivity_analysis.png",
        show=show
    )
    
    # 4. Action gradient components
    print("  - Creating action gradient components...")
    plot_action_gradient_components(
        model, states,
        save_path=output_dir / "action_gradient_components.png",
        show=show
    )
    
    # 5. Gradient comparison
    print("  - Creating gradient comparison...")
    plot_gradient_comparison(
        model, states, state_labels,
        save_path=output_dir / "gradient_comparison.png",
        show=show
    )
    
    # 6. Value gradient
    print("  - Creating value gradient plot...")
    plot_value_gradient(
        model, states,
        save_path=output_dir / "value_gradient.png",
        show=show
    )
    
    print(f"Visualization report complete! Files saved to {output_dir}")


# Set default style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

