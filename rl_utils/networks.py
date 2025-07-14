"""Common networks used in RL.

This file contains nn.Module definitions for common networks used in RL. It is divided into three sets:

1) Common Networks: MLP
2) Common RL Networks:
    For discrete action spaces: DiscreteCritic is a Q-function
    For continuous action spaces: Critic, ValueCritic, and Policy provide the Q-function, value function, and policy respectively.
    For ensembling: ensemblize() provides a wrapper for creating ensembles of networks (e.g. for min-Q / double-Q)
3) Meta Networks for vision tasks:
    WithEncoder: Combines a fully connected network with an encoder network (encoder may come from jaxrl_m.vision)
    ActorCritic: Same as WithEncoder, but for possibly many different networks (e.g. actor, critic, value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform, Independent
import math
from typing import *
from collections import OrderedDict

###############################
#
#  Common Networks
#
###############################


def default_init(scale: Optional[float] = 1.0):
    def _init(layer):
        if isinstance(layer, nn.Linear):
            fan_in = layer.weight.size(1)
            fan_out = layer.weight.size(0)
            scale_val = scale / max(1.0, (fan_in + fan_out) / 2.0)
            limit = math.sqrt(3.0 * scale_val)
            nn.init.uniform_(layer.weight, -limit, limit)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return _init

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int],
                 activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
                 activate_final: bool = False,
                 init_scale: float = 1.0):
        super().__init__()
        self.activations = activations
        self.activate_final = activate_final
        
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i+1])
            layers.append((f"linear_{i}", layer))
            if i < len(dims) - 2 or self.activate_final:
                layers.append((f"activation_{i}", activations()))
        
        self.net = nn.Sequential(OrderedDict(layers))
        self.apply(default_init(init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

###############################
#
#
#  Common RL Networks
#
###############################

class DiscreteCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], n_actions: int,
                 activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU):
        super().__init__()
        self.q_net = MLP(
            input_dim,
            [*hidden_dims, n_actions],
            activations=activations,
            activate_final=False
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.q_net(observations)

class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int],
                 activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.q_net = MLP(
            input_dim,
            [*hidden_dims, 1],
            activations=activations,
            activate_final=False
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([observations, actions], dim=-1)
        return self.q_net(inputs).squeeze(-1)

def ensemblize(cls: Type[nn.Module], num_qs: int, **kwargs) -> nn.Module:
    class Ensemble(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_nets = nn.ModuleList([
                cls(**kwargs) for _ in range(num_qs)
            ])
        
        def forward(self, *args):
            return torch.stack([
                q_net(*args) for q_net in self.q_nets
            ], dim=0)
    
    return Ensemble()

class ValueCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int],
                 activations: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU):
        super().__init__()
        self.value_net = MLP(
            input_dim,
            [*hidden_dims, 1],
            activations=activations,
            activate_final=False
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_net(observations).squeeze(-1)

class Policy(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], action_dim: int,
                 log_std_min: Optional[float] = -20, log_std_max: Optional[float] = 2,
                 tanh_squash_distribution: bool = False,
                 state_dependent_std: bool = True,
                 final_fc_init_scale: float = 1e-2):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash_distribution
        self.state_dependent_std = state_dependent_std
        
        # Shared trunk
        self.trunk = MLP(
            input_dim,
            hidden_dims,
            activations=nn.ReLU,
            activate_final=True
        )
        
        # Output heads
        trunk_output_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.mean_head = nn.Linear(trunk_output_dim, action_dim)
        self.log_std_head = nn.Linear(trunk_output_dim, action_dim) if state_dependent_std else None
        
        # Initialize output layers
        self.mean_head.apply(default_init(final_fc_init_scale))
        if self.log_std_head:
            self.log_std_head.apply(default_init(final_fc_init_scale))
        else:
            self.log_std_param = nn.Parameter(torch.zeros(action_dim))

    def forward(self, observations: torch.Tensor, temperature: float = 1.0) -> torch.distributions.Distribution:
        features = self.trunk(observations)
        
        # Get mean and std
        mean = self.mean_head(features)
        if self.state_dependent_std:
            log_std = self.log_std_head(features)
        else:
            log_std = self.log_std_param.expand_as(mean)
        
        # Constrain log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std) * temperature
        
        # Create distribution
        base_dist = Independent(Normal(mean, std), 1)
        
        # Apply tanh if needed
        if self.tanh_squash:
            transforms = TanhTransform()
            dist = TransformedDistribution(base_dist, transforms)
            dist.mode = lambda: transforms(base_dist.mode)
        else:
            dist = base_dist
            dist.mode = dist.mean
        
        return dist

###############################
#
#
#   Meta Networks for Encoders
#
###############################

def get_latent(encoder: Optional[nn.Module], observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
    if encoder is None:
        return observations
    elif isinstance(observations, dict):
        image_latent = encoder(observations["image"])
        state_latent = observations["state"]
        # Handle different dimensions
        if image_latent.dim() == state_latent.dim():
            return torch.cat([image_latent, state_latent], dim=-1)
        else:
            raise ValueError("Dimension mismatch between image and state latents")
    else:
        return encoder(observations)

class WithEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, network: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.network = network
    
    def forward(self, observations, *args, **kwargs):
        latents = get_latent(self.encoder, observations)
        return self.network(latents, *args, **kwargs)

class ActorCritic(nn.Module):
    """Combines FC networks with encoders for actor, critic, and value.

    Note: You can share encoder parameters between actor and critic by passing in the same encoder definition for both.

    Example:

        encoder_def = ImpalaEncoder()
        actor_def = Policy(...)
        critic_def = Critic(...)
        # This will share the encoder between actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': encoder_def},
            networks={'actor': actor_def, 'critic': critic_def}
        )
        # This will have separate encoders for actor and critic
        model_def = ActorCritic(
            encoders={'actor': encoder_def, 'critic': copy.deepcopy(encoder_def)},
            networks={'actor': actor_def, 'critic': critic_def}
        )
    """
    def __init__(self, encoders: Dict[str, nn.Module], networks: Dict[str, nn.Module]):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.networks = nn.ModuleDict(networks)
    
    def actor(self, observations, **kwargs):
        latents = get_latent(self.encoders["actor"], observations)
        return self.networks["actor"](latents, **kwargs)
    
    def critic(self, observations, actions, **kwargs):
        latents = get_latent(self.encoders["critic"], observations)
        return self.networks["critic"](latents, actions, **kwargs)
    
    def value(self, observations, **kwargs):
        latents = get_latent(self.encoders["value"], observations)
        return self.networks["value"](latents, **kwargs)
    
    def forward(self, observations, actions):
        outputs = {}
        if "actor" in self.networks:
            outputs["actor"] = self.actor(observations)
        if "critic" in self.networks:
            outputs["critic"] = self.critic(observations, actions)
        if "value" in self.networks:
            outputs["value"] = self.value(observations)
        return outputs

