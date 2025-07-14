import numpy as np
import torch
from typing import Any, Dict, Optional, Sequence, Union
import collections

# Helper functions for nested data structures
def tree_map(func, data):
    """Recursively apply function to all tensors/arrays in a nested structure"""
    if isinstance(data, dict):
        return {k: tree_map(func, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(tree_map(func, v) for v in data)
    else:
        return func(data)

def tree_leaves(data):
    """Flatten nested structure into a list of leaf values"""
    if isinstance(data, dict):
        return [leaf for v in data.values() for leaf in tree_leaves(v)]
    elif isinstance(data, (list, tuple)):
        return [leaf for v in data for leaf in tree_leaves(v)]
    else:
        return [data]

def get_size(data: Dict) -> int:
    """Get batch dimension size from nested structure"""
    leaves = tree_leaves(data)
    if not leaves:
        return 0
    sizes = [leaf.shape[0] for leaf in leaves if isinstance(leaf, (np.ndarray, torch.Tensor))]
    return max(sizes) if sizes else 0

class Dataset:
    """Stores and samples batches from nested data dictionary structure"""
    def __init__(self, data: Dict):
        self.data = data
        self.size = get_size(data)

    @classmethod
    def create(
        cls,
        observations: Dict,
        actions: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
        masks: Union[np.ndarray, torch.Tensor],
        next_observations: Dict,
        **extra_fields
    ):
        """Create dataset from components"""
        data = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
            "next_observations": next_observations,
            **extra_fields,
        }
        # Convert all numpy arrays to PyTorch tensors
        data = tree_map(lambda x: torch.as_tensor(x) if isinstance(x, np.ndarray) else x, data)
        return cls(data)

    def sample(self, batch_size: int, indx: Optional[np.ndarray] = None) -> Dict:
        """Sample batch of data, random indices by default"""
        if indx is None:
            indx = np.random.randint(self.size, size=batch_size)
        return self.get_subset(indx)

    def get_subset(self, indices: np.ndarray) -> Dict:
        """Retrieve specific indices from dataset"""
        return tree_map(lambda arr: arr[indices], self.data)


class ReplayBuffer(Dataset):
    """Buffer for storing and sampling transitions with dynamic updates"""
    def __init__(self, data: Dict):
        super().__init__(data)
        # Buffer capacity (max_size)
        self.max_size = get_size(data)  
        # Current number of transitions
        self.size = 0
        # Pointer to next writing position
        self.pointer = 0

    @classmethod
    def create(cls, transition: Dict, size: int):
        """Initialize buffer with specified capacity using example transition"""
        def create_buffer(example):
            # Convert to tensor if not already
            example_tensor = torch.as_tensor(example) if not isinstance(example, torch.Tensor) else example
            # Pre-allocate buffer tensor
            return torch.zeros((size, *example_tensor.shape), 
                               dtype=example_tensor.dtype)

        buffer_dict = tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset: Dict, size: int):
        """Initialize buffer from existing dataset"""
        def create_buffer(init_buffer):
            # Pre-allocate buffer tensor
            buffer = torch.zeros((size, *init_buffer.shape[1:]), 
                                 dtype=init_buffer.dtype)
            # Copy initial data
            num_items = min(len(init_buffer), size)
            buffer[:num_items] = init_buffer[:num_items]
            return buffer

        buffer_dict = tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = get_size(init_dataset)
        dataset.pointer = dataset.size % size  # Wrap around if full
        return dataset

    def add_transition(self, transition: Dict):
        """Add a new transition to the buffer"""
        # Ensure transition values are tensors
        transition = tree_map(lambda x: torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x, transition)
        
        # Update buffer at current pointer position
        def assign_value(buffer_tensor, new_value):
            buffer_tensor[self.pointer] = new_value

        # Apply assignment to all data fields
        tree_map(assign_value, self.data, transition)
        
        # Update buffer state
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, indx: Optional[np.ndarray] = None) -> Dict:
        """Sample from stored transitions only (not entire buffer)"""
        if indx is None:
            # Only sample from valid stored region
            if self.size == 0:
                raise RuntimeError("Cannot sample from empty buffer")
            indx = np.random.randint(0, self.size, size=batch_size)
        return super().get_subset(indx)