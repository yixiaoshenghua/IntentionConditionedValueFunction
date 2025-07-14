import torch
import torch.nn as nn
from torch.distributions import Distribution
from typing import Optional, Callable, Any, Dict, Tuple, Union, Sequence
import math
from collections import OrderedDict
import functools

###############################
# Common utilities
###############################

def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])

def update_exponential_moving_average(target, source, alpha):
    """Update target network parameters using exponential moving average"""
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)
    return target


def shard_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Split batch across devices for data parallelism. Analogous to JAX shard_batch.
    Note: Actual device distribution handled by PyTorch's parallel APIs (not implemented here)
    """
    # In practice, use PyTorch Distributed or DataParallel for actual sharding
    return batch  # Placeholder implementation


# def target_update(model: TrainState, 
#                   target_model: TrainState, 
#                   tau: float) -> TrainState:
#     """Update target network parameters using exponential moving average"""
#     with torch.no_grad():
#         for param, target_param in zip(model.model.parameters(), target_model.model.parameters()):
#             target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
#     return target_model

###############################
# TrainState Implementation
###############################

class TrainState:
    """
    PyTorch equivalent of JAX TrainState with similar functionality.
    
    Core differences:
    1. Optimizer state is managed by PyTorch's optimizer
    2. Parameter updates are in-place rather than functional
    3. Gradient computation uses PyTorch autograd
    
    Usage:
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    state = TrainState.create(model, optimizer)
    
    # Forward pass
    output = state(torch.randn(1, 10))
    
    # Update
    loss = output.pow(2).mean()
    next_state, aux = state.apply_loss_fn(lambda m, x: m(x).pow(2).mean(), inputs, has_aux=True)
    """
    
    def __init__(self, 
                 step: int,
                 model: nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 grad_enabled: bool = False):
        self.step = step
        self.model = model
        self.optimizer = optimizer
        self.grad_enabled = grad_enabled
        
    @classmethod
    def create(cls,
               model_def: nn.Module,
               optimizer: Optional[torch.optim.Optimizer] = None,
               **kwargs) -> 'TrainState':
        """Create training state from model definition and optimizer"""
        return cls(step=1, model=model_def, optimizer=optimizer, **kwargs)
    
    def __call__(self,
                 *args,
                 params=None,
                 method: Optional[Union[str, Callable]] = None,
                 **kwargs) -> Any:
        """
        Run forward pass. When `params` is provided, temporarily uses those parameters.
        `method`: Model method name to call, or custom function (model instance is 1st arg)
        """
        # Create context for temporary parameter substitution
        ctx = TemporaryParams(self.model, params) if params else nullcontext()
        
        with ctx, torch.set_grad_enabled(self.grad_enabled):
            if isinstance(method, str):
                method = getattr(self.model, method)
            return method(self.model, *args, **kwargs) if callable(method) else self.model(*args, **kwargs)

    def apply_gradients(self, 
                        grads: Dict[str, torch.Tensor], 
                        **kwargs) -> 'TrainState':
        """
        Apply calculated gradients to update model parameters.
        `grads`: Dict of parameter_name: gradient_tensor pairs
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not defined for apply_gradients")
        
        # Assign gradients
        for name, param in self.model.named_parameters():
            if name in grads:
                if param.grad is None:
                    param.grad = grads[name].detach().clone()
                else:
                    param.grad.copy_(grads[name])
        
        # Update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        
        return self

    def apply_loss_fn(self,
                      loss_fn: Callable[[nn.Module, Any], Any],
                      *args,
                      pmap_axis: Optional[str] = None,
                      has_aux: bool = False,
                      **kwargs) -> Union['TrainState', Tuple['TrainState', Any]]:
        """
        Compute loss and apply gradients through backpropagation.
        `pmap_axis`: Placeholder for compatibility (use PyTorch parallel APIs)
        """
        # Forward pass with gradient computation
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        outputs = loss_fn(self.model, *args, **kwargs)
        
        if has_aux:
            loss, aux = outputs
        else:
            loss = outputs
            
        # Backpropagation
        loss.backward()
        
        # Apply gradients
        self.optimizer.step()
        self.step += 1
        
        if has_aux:
            return self, aux
        return self


###############################
# Helper Utilities
###############################

class TemporaryParams:
    """Context manager for temporary parameter substitution"""
    def __init__(self, model: nn.Module, params: Dict[str, torch.Tensor]):
        self.model = model
        self.temp_params = params
        self.original_params = {}
        
    def __enter__(self):
        # Save original parameters and replace with temp ones
        for name, param in self.model.named_parameters():
            if name in self.temp_params:
                self.original_params[name] = param.data.clone()
                param.data.copy_(self.temp_params[name])
                
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original parameters
        for name, param in self.model.named_parameters():
            if name in self.original_params:
                param.data.copy_(self.original_params[name])

class nullcontext:
    """Placeholder context manager"""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass