import torch
import dataclasses
from typing import Any, Dict, Optional

@dataclasses.dataclass
class GCDataset:
    dataset: Dict[str, torch.Tensor]  # Dataset as dictionary of tensors
    p_randomgoal: float              # Probability to sample random goal
    p_trajgoal: float               # Probability to sample future trajectory goal
    p_currgoal: float               # Probability to use current state as goal
    terminal_key: str = 'dones_float'  # Key for termination flags
    reward_scale: float = 1.0       # Reward scaling factor
    reward_shift: float = -1.0      # Reward shift factor
    terminal: bool = True           # Whether to terminate on success
    max_distance: Optional[int] = None  # Max steps away for trajectory goals
    curr_goal_shift: int = 0        # Index offset for current goal

    def __post_init__(self):
        """Identify terminal state indices and validate probability weights."""
        # Get indices where episode terminates
        self.terminal_locs = torch.nonzero(
            self.dataset[self.terminal_key] > 0.5
        ).squeeze()
        
        # Validate probability distribution
        assert (self.p_randomgoal + self.p_trajgoal + self.p_currgoal - 1.0).abs() < 1e-6, \
            "Goal sampling probabilities must sum to 1"

    def sample_goals(self, indx: torch.Tensor, 
                    p_randomgoal: Optional[float] = None,
                    p_trajgoal: Optional[float] = None,
                    p_currgoal: Optional[float] = None) -> torch.Tensor:
        """
        Sample goals using probabilistic strategy:
        - p_randomgoal: uniformly random goals
        - p_trajgoal: goals from same trajectory
        - p_currgoal: current state goals
        """
        # Use default probabilities if not specified
        p_randomgoal = self.p_randomgoal if p_randomgoal is None else p_randomgoal
        p_trajgoal = self.p_trajgoal if p_trajgoal is None else p_trajgoal
        p_currgoal = self.p_currgoal if p_currgoal is None else p_currgoal
        
        batch_size, device = len(indx), indx.device
        
        # 1. Random goals (uniform sampling)
        goal_indx = torch.randint(
            0, len(self.dataset['observations']) - self.curr_goal_shift,
            (batch_size,), device=device
        )
        
        # 2. Trajectory goals (interpolate between current and terminal state)
        # Find terminal state for each index
        traj_idx = torch.bucketize(indx, self.terminal_locs)
        traj_idx = torch.clamp(traj_idx, max=len(self.terminal_locs)-1)
        final_state_indx = self.terminal_locs[traj_idx]
        
        # Apply max distance constraint if specified
        if self.max_distance is not None:
            final_state_indx = torch.min(
                final_state_indx, indx + self.max_distance
            )
        
        # Interpolate intermediate goal
        distance = torch.rand(batch_size, device=device)
        middle_goal_indx = torch.round(
            indx.float() * (1 - distance) + final_state_indx.float() * distance
        ).long()
        
        # Select between trajectory and random goals
        traj_mask = torch.rand(batch_size, device=device) < (
            p_trajgoal / (p_trajgoal + p_randomgoal + 1e-8)
        )
        goal_indx = torch.where(traj_mask, middle_goal_indx, goal_indx)
        
        # 3. Current state goals
        curr_mask = torch.rand(batch_size, device=device) < p_currgoal
        goal_indx = torch.where(curr_mask, indx, goal_indx)
        
        return goal_indx

    def sample(self, batch_size: int, 
              indx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Sample batch of transitions with goal relabeling.
        Returns dictionary containing observations, goals, rewards, etc.
        """
        # Sample transition indices if not provided
        if indx is None:
            indx = torch.randint(
                0, len(self.dataset['observations']) - 1,
                (batch_size,), device='cpu'
            ).to(self.dataset['observations'].device)
        
        # Get basic transitions
        batch = {k: v[indx] for k, v in self.dataset.items()}
        
        # Sample goals and calculate success
        goal_indx = self.sample_goals(indx)
        success = (indx == goal_indx).float()
        
        # Compute rewards and termination masks
        batch['rewards'] = success * self.reward_scale + self.reward_shift
        batch['masks'] = 1.0 - success if self.terminal else torch.ones_like(success)
        
        # Adjust goal indices with shift and bounds
        goal_indx = torch.clamp(
            goal_indx + self.curr_goal_shift,
            0, len(self.dataset['observations']) - 1
        )
        
        # Add goals to batch (assumes observations is a tensor dict)
        batch['goals'] = {
            k: v[goal_indx] for k, v in self.dataset['observations'].items()
        }
        
        return batch

@dataclasses.dataclass
class GCSDataset(GCDataset):
    p_samegoal: float = 0.5          # Probability to share goals between streams
    intent_sametraj: bool = False   # Whether to force intent goals in same trajectory

    def sample(self, batch_size: int, 
              indx: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Sample batch with dual goal conditioning:
        - goals: primary goals for policy optimization
        - desired_goals: secondary goals for skill conditioning
        """
        if indx is None:
            indx = torch.randint(
                0, len(self.dataset['observations']) - 1,
                (batch_size,), device='cpu'
            ).to(self.dataset['observations'].device)
        
        # Base transition sampling
        batch = {k: v[indx] for k, v in self.dataset.items()}
        
        # Sample desired goals (intent goals)
        if self.intent_sametraj:
            desired_goal_indx = self.sample_goals(
                indx, p_randomgoal=0.0, 
                p_trajgoal=1.0 - self.p_currgoal,
                p_currgoal=self.p_currgoal
            )
        else:
            desired_goal_indx = self.sample_goals(indx)
        
        # Sample primary goals with possible sharing
        goal_indx = self.sample_goals(indx)
        same_mask = torch.rand(batch_size, device=indx.device) < self.p_samegoal
        goal_indx = torch.where(same_mask, desired_goal_indx, goal_indx)
        
        # Calculate success metrics
        success = (indx == goal_indx).float()
        desired_success = (indx == desired_goal_indx).float()
        
        # Update rewards and masks for both goal types
        batch['rewards'] = success * self.reward_scale + self.reward_shift
        batch['desired_rewards'] = desired_success * self.reward_scale + self.reward_shift
        
        batch['masks'] = 1.0 - success if self.terminal else torch.ones_like(success)
        batch['desired_masks'] = 1.0 - desired_success if self.terminal else torch.ones_like(desired_success)
        
        # Adjust indices with bounds checking
        goal_indx = torch.clamp(
            goal_indx + self.curr_goal_shift,
            0, len(self.dataset['observations']) - 1
        )
        desired_goal_indx = torch.clamp(
            desired_goal_indx + self.curr_goal_shift,
            0, len(self.dataset['observations']) - 1
        )
        
        # Add both goal types to batch
        batch['goals'] = {
            k: v[goal_indx] for k, v in self.dataset['observations'].items()
        }
        batch['desired_goals'] = {
            k: v[desired_goal_indx] for k, v in self.dataset['observations'].items()
        }
        
        return batch