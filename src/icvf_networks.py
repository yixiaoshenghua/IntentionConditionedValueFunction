import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

class LayerNormMLP(nn.Module):
    """
    Multi-layer perceptron with Layer Normalization.
    
    Args:
        hidden_dims: List of hidden layer dimensions
        activations: Activation function (default: GELU)
        activate_final: Whether to apply activation after last layer
        kernel_init: Weight initialization method
    """
    def __init__(self, 
                 hidden_dims: Sequence[int],
                 activations: Callable[[torch.Tensor], torch.Tensor] = nn.GELU(),
                 activate_final: bool = False,
                 kernel_init: Optional[Callable] = None):
        super().__init__()
        layers = []
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.LazyLinear(dim))
            if i < len(hidden_dims) - 1 or activate_final:
                layers.append(activations)
                layers.append(nn.LayerNorm(dim))
        self.net = nn.Sequential(*layers)
        
        # Apply custom initialization if provided
        if kernel_init:
            self.apply(lambda m: kernel_init(m) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ICVFWithEncoder(nn.Module):
    """
    Value function wrapper with encoder module
    
    Args:
        encoder: Feature extractor module
        vf: Value function module
    """
    def __init__(self, 
                 encoder: nn.Module, 
                 vf: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.vf = vf

    def get_encoder_latent(self, observations: torch.Tensor) -> torch.Tensor:
        return self._get_latent(observations)
    
    def get_phi(self, observations: torch.Tensor) -> torch.Tensor:
        latent = self._get_latent(observations)
        return self.vf.get_phi(latent)

    def forward(self, 
                observations: torch.Tensor,
                outcomes: torch.Tensor,
                intents: torch.Tensor) -> torch.Tensor:
        latent_s = self._get_latent(observations)
        latent_g = self._get_latent(outcomes)
        latent_z = self._get_latent(intents)
        return self.vf(latent_s, latent_g, latent_z)
    
    def get_info(self, 
                 observations: torch.Tensor,
                 outcomes: torch.Tensor,
                 intents: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent_s = self._get_latent(observations)
        latent_g = self._get_latent(outcomes)
        latent_z = self._get_latent(intents)
        return self.vf.get_info(latent_s, latent_g, latent_z)
    
    def _get_latent(self, x: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """Unified feature extraction method"""
        if isinstance(x, dict):
            image_latent = self.encoder(x["image"])
            state_latent = x["state"]
            return torch.cat([image_latent, state_latent], dim=-1)
        return self.encoder(x)


class ICVFTemplate(nn.Module):
    """Base class for Implicit Cross Entropy Value Functions"""
    def get_info(self, 
                 observations: torch.Tensor,
                 outcomes: torch.Tensor,
                 intents: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def get_phi(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self,
                observations: torch.Tensor,
                outcomes: torch.Tensor,
                intents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MonolithicVF(ICVFTemplate):
    """
    Single neural network value function implementation
    
    Args:
        hidden_dims: List of hidden layer dimensions
        use_layer_norm: Whether to use layer normalization
    """
    def __init__(self,
                 hidden_dims: Sequence[int],
                 use_layer_norm: bool = False):
        super().__init__()
        network_cls = LayerNormMLP if use_layer_norm else nn.LazyLinear
        self.net = network_cls(hidden_dims + [1])
        self.repr_warning = True

    def get_info(self, 
                 observations: torch.Tensor,
                 outcomes: torch.Tensor,
                 intents: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([observations, outcomes, intents], dim=-1)
        v = self.net(x).squeeze(-1)
        return {'v': v, 'psi': outcomes, 'z': intents, 'phi': observations}
    
    def get_phi(self, observations: torch.Tensor) -> torch.Tensor:
        if self.repr_warning:
            print('Warning: StandardVF returns raw states as representations')
            self.repr_warning = False
        return observations
    
    def forward(self,
                observations: torch.Tensor,
                outcomes: torch.Tensor,
                intents: torch.Tensor) -> torch.Tensor:
        x = torch.cat([observations, outcomes, intents], dim=-1)
        return self.net(x).squeeze(-1)


class MultilinearVF(ICVFTemplate):
    """
    Factorized value function with tensor decomposition
    
    Args:
        hidden_dims: List of hidden layer dimensions
        use_layer_norm: Whether to use layer normalization
    """
    def __init__(self,
                 hidden_dims: Sequence[int],
                 use_layer_norm: bool = False):
        super().__init__()
        network_cls = LayerNormMLP if use_layer_norm else nn.Sequential
        
        # State representation network
        self.phi_net = network_cls(
            nn.LazyLinear(hidden_dims[-1]),
            nn.GELU() if use_layer_norm else None
        )
        
        # Outcome representation network
        self.psi_net = network_cls(
            nn.LazyLinear(hidden_dims[-1]),
            nn.GELU() if use_layer_norm else None
        )
        
        # Intent transformation network
        self.T_net = network_cls(
            nn.LazyLinear(hidden_dims[-1]),
            nn.GELU() if use_layer_norm else None
        )
        
        # Intent-conditioned projection matrices
        self.matrix_a = nn.LazyLinear(hidden_dims[-1])
        self.matrix_b = nn.LazyLinear(hidden_dims[-1])

    def get_phi(self, observations: torch.Tensor) -> torch.Tensor:
        return self.phi_net(observations)

    def get_info(self,
                 observations: torch.Tensor,
                 outcomes: torch.Tensor,
                 intents: torch.Tensor) -> Dict[str, torch.Tensor]:
        phi = self.phi_net(observations)
        psi = self.psi_net(outcomes)
        z = self.psi_net(intents)
        Tz = self.T_net(z)
        
        # Low-rank intent-conditioned projections
        phi_z = self.matrix_a(Tz * phi)
        psi_z = self.matrix_b(Tz * psi)
        
        # Bilinear value function
        v = (phi_z * psi_z).sum(dim=-1)
        
        return {
            'v': v,
            'phi': phi,
            'psi': psi,
            'Tz': Tz,
            'z': z,
            'phi_z': phi_z,
            'psi_z': psi_z
        }
    
    def forward(self,
                observations: torch.Tensor,
                outcomes: torch.Tensor,
                intents: torch.Tensor) -> torch.Tensor:
        return self.get_info(observations, outcomes, intents)['v']


# ICVF type registry
ICVF_REGISTRY = {
    'multilinear': MultilinearVF,
    'monolithic': MonolithicVF
}

def create_icvf(icvf_type: Union[str, ICVFTemplate],
               encoder: Optional[nn.Module] = None,
               ensemble: bool = True,
               **kwargs) -> nn.Module:
    """
    Create ICVF model with optional encoder and ensemble
    
    Args:
        icvf_type: ICVF type name or class
        encoder: Optional feature encoder module
        ensemble: Whether to create ensemble of value functions
        **kwargs: Constructor arguments for ICVF
    """
    # Resolve ICVF type
    if isinstance(icvf_type, str):
        icvf_cls = ICVF_REGISTRY[icvf_type]
    else:
        icvf_cls = icvf_type
    
    # Create value function (single or ensemble)
    if ensemble:
        vf = EnsembleICVF(icvf_cls, 2, **kwargs)
    else:
        vf = icvf_cls(**kwargs)
    
    # Add encoder wrapper if needed
    if encoder:
        return ICVFWithEncoder(encoder, vf)
    return vf


class EnsembleICVF(nn.Module):
    """
    Ensemble of ICVF models for uncertainty estimation
    
    Args:
        model_cls: ICVF model class
        num_models: Number of ensemble members
        **kwargs: Constructor arguments for ICVF
    """
    def __init__(self, 
                 model_cls: ICVFTemplate, 
                 num_models: int,
                 **kwargs):
        super().__init__()
        self.models = nn.ModuleList([model_cls(**kwargs) for _ in range(num_models)])
    
    def forward(self,
                observations: torch.Tensor,
                outcomes: torch.Tensor,
                intents: torch.Tensor) -> torch.Tensor:
        outputs = [model(observations, outcomes, intents) for model in self.models]
        return torch.stack(outputs, dim=0)  # [ensemble, batch]
    
    def get_info(self,
                 observations: torch.Tensor,
                 outcomes: torch.Tensor,
                 intents: torch.Tensor) -> Dict[str, torch.Tensor]:
        infos = [model.get_info(observations, outcomes, intents) for model in self.models]
        
        # Stack all outputs from ensemble
        return {k: torch.stack([info[k] for info in infos], dim=0) for k in infos[0]}
    
    def get_phi(self, observations: torch.Tensor) -> torch.Tensor:
        phis = [model.get_phi(observations) for model in self.models]
        return torch.stack(phis, dim=0)  # [ensemble, batch, features]