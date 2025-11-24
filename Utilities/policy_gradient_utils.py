"""
Utility functions for computing policy gradients and derivatives
with respect to state observations.
"""

import numpy as np
import torch
from typing import Union, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


def compute_policy_gradient(
    model: PPO,
    state: Union[np.ndarray, torch.Tensor],
    wrt_action_mean: bool = True,
    wrt_action_log_std: bool = False,
    wrt_value: bool = False,
) -> dict:
    """
    Compute the gradient of the policy function with respect to a given state.
    
    Args:
        model: Trained PPO model
        state: State/observation (numpy array or torch tensor)
        wrt_action_mean: If True, compute gradient of action mean w.r.t. state
        wrt_action_log_std: If True, compute gradient of action log_std w.r.t. state
        wrt_value: If True, compute gradient of value function w.r.t. state
        
    Returns:
        Dictionary containing gradients:
        - 'action_mean_grad': Gradient of action mean w.r.t. state (if wrt_action_mean=True)
        - 'action_log_std_grad': Gradient of action log_std w.r.t. state (if wrt_action_log_std=True)
        - 'value_grad': Gradient of value w.r.t. state (if wrt_value=True)
        - 'action_mean': The action mean at this state
        - 'value': The value estimate at this state
    """
    # Set policy to evaluation mode
    model.policy.eval()
    
    # Convert state to torch tensor if needed
    if isinstance(state, np.ndarray):
        state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True)
    else:
        state_tensor = state.clone().detach().requires_grad_(True)
    
    # Ensure state is 2D (batch dimension)
    if state_tensor.dim() == 1:
        state_tensor = state_tensor.unsqueeze(0)
    
    results = {}
    
    # Forward pass to get values (without gradients)
    with torch.no_grad():
        features = model.policy.extract_features(state_tensor)
        latent_pi = model.policy.mlp_extractor.forward_actor(features)
        latent_vf = model.policy.mlp_extractor.forward_critic(features)
        mean_actions_no_grad = model.policy.action_net(latent_pi)
        values_no_grad = model.policy.value_net(latent_vf)
        results['action_mean'] = mean_actions_no_grad.detach().numpy()
        results['value'] = values_no_grad.detach().numpy()
    
    # Now compute gradients with fresh computation graph
    # Convert state to numpy first to ensure we create a fresh tensor
    if isinstance(state, torch.Tensor):
        state_np = state.detach().cpu().numpy()
    else:
        state_np = np.array(state)
    
    # Ensure state is 2D (batch dimension)
    if len(state_np.shape) == 1:
        state_np = state_np[np.newaxis, :]
    
    # Create a fresh leaf tensor with requires_grad=True
    # Create directly in the right shape to avoid unsqueeze creating non-leaf tensor
    state_tensor_grad = torch.tensor(state_np, dtype=torch.float32, requires_grad=True)
    
    features = model.policy.extract_features(state_tensor_grad)
    latent_pi = model.policy.mlp_extractor.forward_actor(features)
    latent_vf = model.policy.mlp_extractor.forward_critic(features)
    
    if wrt_action_mean:
        mean_actions = model.policy.action_net(latent_pi)
        action_dim = mean_actions.shape[1]
        action_mean_grads = []
        
        # Compute gradient for each action dimension
        for i in range(action_dim):
            # Zero out gradients from previous iteration
            if state_tensor_grad.grad is not None:
                state_tensor_grad.grad.zero_()
            else:
                # Ensure gradients are enabled
                state_tensor_grad.requires_grad_(True)
            
            # Compute gradient of action[i] w.r.t. state
            mean_actions[0, i].backward(retain_graph=(i < action_dim - 1))
            
            # Check if gradient was computed
            if state_tensor_grad.grad is None:
                raise RuntimeError(f"Gradient computation failed for action dimension {i}. "
                                 f"state_tensor_grad.requires_grad={state_tensor_grad.requires_grad}, "
                                 f"is_leaf={state_tensor_grad.is_leaf}")
            
            action_mean_grads.append(state_tensor_grad.grad[0].clone())
        
        # Stack gradients: shape will be (action_dim, state_dim)
        results['action_mean_grad'] = torch.stack(action_mean_grads, dim=0).detach().numpy()
    
    if wrt_value:
        # If we computed action_mean gradients, the computation graph may have been consumed
        # So we need to recompute the forward pass
        if wrt_action_mean:
            # Recompute forward pass for value
            features = model.policy.extract_features(state_tensor_grad)
            latent_vf = model.policy.mlp_extractor.forward_critic(features)
        
        # Zero out gradients if we computed action_mean gradients
        if state_tensor_grad.grad is not None:
            state_tensor_grad.grad.zero_()
        else:
            # Ensure gradients are enabled
            state_tensor_grad.requires_grad_(True)
        
        values = model.policy.value_net(latent_vf)
        values[0, 0].backward()
        
        # Check if gradient was computed
        if state_tensor_grad.grad is None:
            raise RuntimeError("Gradient computation failed for value function. "
                             f"state_tensor_grad.requires_grad={state_tensor_grad.requires_grad}, "
                             f"is_leaf={state_tensor_grad.is_leaf}")
        
        results['value_grad'] = state_tensor_grad.grad[0].clone().detach().numpy()
    
    return results


def compute_policy_jacobian(
    model: PPO,
    state: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """
    Compute the Jacobian matrix of the policy (action mean) with respect to state.
    
    The Jacobian has shape (action_dim, state_dim), where element [i, j] is
    the partial derivative of action[i] with respect to state[j].
    
    This is equivalent to calling compute_policy_gradient with wrt_action_mean=True
    and extracting the 'action_mean_grad' field.
    
    Args:
        model: Trained PPO model
        state: State/observation (numpy array or torch tensor)
        
    Returns:
        Jacobian matrix of shape (action_dim, state_dim)
    """
    grad_info = compute_policy_gradient(model, state, wrt_action_mean=True)
    return grad_info['action_mean_grad']


def get_policy_sensitivity(
    model: PPO,
    state: Union[np.ndarray, torch.Tensor],
    state_perturbation: Optional[Union[np.ndarray, torch.Tensor]] = None,
    epsilon: float = 1e-5,
) -> dict:
    """
    Compute policy sensitivity: how much the action changes when state is perturbed.
    
    This can be done either:
    1. Numerically (finite differences) if state_perturbation is provided
    2. Analytically (using gradients) if state_perturbation is None
    
    Args:
        model: Trained PPO model
        state: Base state/observation
        state_perturbation: Optional perturbation vector. If None, uses gradient.
        epsilon: Small value for numerical differentiation
        
    Returns:
        Dictionary with sensitivity information:
        - 'action_at_state': Action at base state
        - 'action_change': Change in action (if perturbation provided)
        - 'gradient': Gradient of action w.r.t. state
        - 'sensitivity_norm': Norm of the gradient
    """
    # Get action at base state
    if isinstance(state, torch.Tensor):
        state_np = state.detach().numpy()
    else:
        state_np = np.array(state)
    
    action_base, _ = model.predict(state_np, deterministic=True)
    
    results = {
        'action_at_state': action_base,
    }
    
    # Compute gradient
    grad_info = compute_policy_gradient(model, state, wrt_action_mean=True)
    results['gradient'] = grad_info['action_mean_grad']
    results['sensitivity_norm'] = np.linalg.norm(results['gradient'])
    
    # If perturbation provided, also compute numerical sensitivity
    if state_perturbation is not None:
        if isinstance(state_perturbation, torch.Tensor):
            pert_np = state_perturbation.detach().numpy()
        else:
            pert_np = np.array(state_perturbation)
        
        state_perturbed = state_np + epsilon * pert_np
        action_perturbed, _ = model.predict(state_perturbed, deterministic=True)
        results['action_change'] = action_perturbed - action_base
        results['numerical_sensitivity'] = results['action_change'] / epsilon
    
    return results

