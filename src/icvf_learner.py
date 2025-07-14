import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Any, Callable, Optional
from collections import OrderedDict

def expectile_loss(adv: torch.Tensor, 
                  diff: torch.Tensor, 
                  expectile: float = 0.8) -> torch.Tensor:
    """
    Compute expectile loss for value function training
    
    Args:
        adv: Advantage tensor
        diff: TD error tensor (Q - V)
        expectile: Expectile parameter controlling asymmetry
    """
    weight = torch.where(adv >= 0, 
                         torch.tensor(expectile, device=adv.device),
                         torch.tensor(1 - expectile, device=adv.device))
    return weight * diff ** 2

def icvf_loss(value_net: nn.Module,
              target_value_net: nn.Module,
              batch: Dict[str, torch.Tensor],
              config: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute ICVF loss for implicit cross-entropy value function
    
    Args:
        value_net: Current value network
        target_value_net: Target value network
        batch: Dictionary containing:
            observations, next_observations, goals, desired_goals,
            rewards, masks, desired_rewards, desired_masks
        config: Configuration dictionary with keys:
            no_intent, min_q, expectile, discount
    """
    # Handle no_intent case
    if config['no_intent']:
        batch['desired_goals'] = torch.ones_like(batch['desired_goals'])
    
    # Compute TD error for outcome s_+
    with torch.no_grad():
        next_v1_gz, next_v2_gz = target_value_net(
            batch['next_observations'], 
            batch['goals'], 
            batch['desired_goals']
        )
        q1_gz = batch['rewards'] + config['discount'] * batch['masks'] * next_v1_gz
        q2_gz = batch['rewards'] + config['discount'] * batch['masks'] * next_v2_gz
    
    v1_gz, v2_gz = value_net(
        batch['observations'], 
        batch['goals'], 
        batch['desired_goals']
    )
    
    # Compute advantage under intent z
    with torch.no_grad():
        next_v1_zz, next_v2_zz = target_value_net(
            batch['next_observations'], 
            batch['desired_goals'], 
            batch['desired_goals']
        )
        if config['min_q']:
            next_v_zz = torch.min(next_v1_zz, next_v2_zz)
        else:
            next_v_zz = (next_v1_zz + next_v2_zz) / 2
        
        q_zz = batch['desired_rewards'] + config['discount'] * batch['desired_masks'] * next_v_zz
        
        v1_zz, v2_zz = target_value_net(
            batch['observations'], 
            batch['desired_goals'], 
            batch['desired_goals']
        )
        v_zz = (v1_zz + v2_zz) / 2
        adv = q_zz - v_zz
        
        if config['no_intent']:
            adv = torch.zeros_like(adv)
    
    # Compute value loss with expectile weighting
    value_loss1 = expectile_loss(adv, q1_gz - v1_gz, config['expectile']).mean()
    value_loss2 = expectile_loss(adv, q2_gz - v2_gz, config['expectile']).mean()
    value_loss = value_loss1 + value_loss2
    
    # Helper for masked mean calculation
    def masked_mean(x, mask):
        return (x * mask).sum() / (mask.sum() + 1e-5)
    
    # Metrics collection
    metrics = {
        'value_loss': value_loss,
        'v_gz_max': v1_gz.max(),
        'v_gz_min': v1_gz.min(),
        'v_zz': v_zz.mean(),
        'v_gz': v1_gz.mean(),
        'abs_adv_mean': torch.abs(adv).mean(),
        'adv_mean': adv.mean(),
        'adv_max': adv.max(),
        'adv_min': adv.min(),
        'accept_prob': (adv >= 0).float().mean(),
        'reward_mean': batch['rewards'].mean(),
        'mask_mean': batch['masks'].mean(),
        'q_gz_max': q1_gz.max(),
        'value_loss1': masked_mean((q1_gz - v1_gz)**2, batch['masks']),
        'value_loss2': masked_mean((q1_gz - v1_gz)**2, 1.0 - batch['masks'])
    }
    
    return value_loss, metrics

def periodic_target_update(model: nn.Module, 
                           target_model: nn.Module, 
                           step: int, 
                           period: int) -> None:
    """
    Update target network parameters periodically
    
    Args:
        model: Current model
        target_model: Target model
        step: Current training step
        period: Update period
    """
    if step % period == 0:
        target_model.load_state_dict(model.state_dict())

class ICVFAgent(nn.Module):
    """
    Implicit Cross Entropy Value Function Agent
    
    Args:
        value_net: Value network module
        target_value_net: Target value network module
        optimizer: Optimizer for value network
        config: Configuration dictionary
    """
    def __init__(self,
                 value_net: nn.Module,
                 target_value_net: nn.Module,
                 optimizer: optim.Optimizer,
                 config: Dict[str, Any]):
        super().__init__()
        self.value_net = value_net
        self.target_value_net = target_value_net
        self.optimizer = optimizer
        self.config = config
        self.step = 0
        
        # Initialize target network with same weights
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_value_net.eval()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Perform a single update step
        
        Args:
            batch: Training batch dictionary
        Returns:
            Tuple of (agent state, metrics)
        """
        self.value_net.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        loss, metrics = icvf_loss(
            self.value_net,
            self.target_value_net,
            batch,
            self.config
        )
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.config['periodic_target_update']:
            periodic_target_update(
                self.value_net,
                self.target_value_net,
                self.step,
                int(1.0 / self.config['target_update_rate'])
            )
        else:
            # Soft update
            for target_param, param in zip(self.target_value_net.parameters(), 
                                          self.value_net.parameters()):
                target_param.data.copy_(
                    self.config['target_update_rate'] * param.data +
                    (1 - self.config['target_update_rate']) * target_param.data
                )
        
        self.step += 1
        return self.state_dict(), metrics

def create_learner(
    seed: int,
    value_net: nn.Module,
    optim_kwargs: Dict[str, Any] = {
        'lr': 0.00005,
        'eps': 0.0003125
    },
    discount: float = 0.95,
    target_update_rate: float = 0.005,
    expectile: float = 0.9,
    no_intent: bool = False,
    min_q: bool = True,
    periodic_target_update: bool = False,
    **kwargs
) -> ICVFAgent:
    """
    Create ICVF learner agent
    
    Args:
        seed: Random seed
        value_net: Value network module
        optim_kwargs: Optimizer parameters
        discount: Discount factor
        target_update_rate: Target network update rate
        expectile: Expectile parameter for loss
        no_intent: Whether to ignore intent
        min_q: Whether to use min of two Q-values
        periodic_target_update: Use periodic instead of soft updates
        **kwargs: Additional configuration parameters
    """
    # Set random seed
    torch.manual_seed(seed)
    
    # Create optimizer
    optimizer = optim.Adam(value_net.parameters(), **optim_kwargs)
    
    # Create target network
    target_value_net = type(value_net)()
    target_value_net.load_state_dict(value_net.state_dict())
    target_value_net.eval()
    
    # Create configuration
    config = {
        'discount': discount,
        'target_update_rate': target_update_rate,
        'expectile': expectile,
        'no_intent': no_intent,
        'min_q': min_q,
        'periodic_target_update': periodic_target_update,
        **kwargs
    }
    
    return ICVFAgent(value_net, target_value_net, optimizer, config)

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for ICVF agent
    """
    return {
        'optim_kwargs': {
            'lr': 0.00005,
            'eps': 0.0003125
        },
        'discount': 0.99,
        'expectile': 0.9,
        'target_update_rate': 0.005,
        'no_intent': False,
        'min_q': True,
        'periodic_target_update': False
    }
