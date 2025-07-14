import sys
import os
import pathlib
sys.path.append(str(pathlib.Path(os.path.abspath(os.path.dirname(__file__))).parent))
sys.path.append(str(pathlib.Path(os.path.abspath(os.path.dirname(__file__))).parent.parent))
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import functools
from collections import OrderedDict

# Import your PyTorch implementations of these modules
from src import icvf_learner as learner
from src.icvf_networks import ICVF_REGISTRY, create_icvf
from icvf_envs.antmaze import d4rl_utils, d4rl_ant, d4rl_pm
from src.gc_dataset import GCSDataset
from src import viz_utils

class DebugPlotGenerator:
    def __init__(self, env_name, gc_dataset):
        self.env_name = env_name
        if 'antmaze' in env_name:
            viz_env, viz_dataset = d4rl_ant.get_env_and_dataset(env_name)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (12.5, 8)
            viz_library = d4rl_ant
            self.viz_things = (viz_env, viz_dataset, viz_library, init_state)
        elif 'maze' in env_name:
            viz_env, viz_dataset = d4rl_pm.get_gcenv_and_dataset(env_name)
            init_state = np.copy(viz_dataset['observations'][0])
            init_state[:2] = (3, 4)
            viz_library = d4rl_pm
            self.viz_things = (viz_env, viz_dataset, viz_library, init_state)
        else:
            raise NotImplementedError('Visualization not implemented for this environment')
        
        # Intent selection
        intent_set_indx = torch.as_tensor(np.array([184588, 62200, 162996, 110214, 4086, 191369, 92549, 12946, 192021]))
        self.intent_set_batch = gc_dataset.sample(9, indx=intent_set_indx)
        self.example_trajectory = gc_dataset.sample(50, indx=torch.arange(1000, 1050))

    def generate_debug_plots(self, agent):
        example_trajectory = self.example_trajectory
        intents = self.intent_set_batch['observations']
        (viz_env, viz_dataset, viz_library, init_state) = self.viz_things
        
        visualizations = {}
        traj_metrics = get_traj_v(agent, example_trajectory)
        
        # Create visualizations (implementation depends on your viz_utils)
        value_viz = viz_utils.make_visual_no_image(traj_metrics, [
            functools.partial(viz_utils.visualize_metric, metric_name=k) 
            for k in traj_metrics.keys()
        ])
        visualizations['value_traj_viz'] = value_viz
        
        if 'maze' in self.env_name:
            print('Visualizing intent policies and values')
            # Policy visualization
            methods = [
                functools.partial(viz_library.plot_policy, policy_fn=functools.partial(get_policy, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            policy_images = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_policies'] = policy_images
            
            # Value visualization
            methods = [
                functools.partial(viz_library.plot_value, value_fn=functools.partial(get_values, agent, intent=intents[idx]))
                for idx in range(9)
            ]
            value_images = viz_library.make_visual(viz_env, viz_dataset, methods)
            visualizations['intent_values'] = value_images
            
            # Combined visualizations
            for idx in range(3):
                policy_fn = functools.partial(get_policy, agent, intent=intents[idx])
                value_fn = functools.partial(get_values, agent, intent=intents[idx])
                image = viz_library.make_visual(viz_env, viz_dataset, [
                    functools.partial(viz_library.plot_policy, policy_fn=policy_fn),
                    functools.partial(viz_library.plot_value, value_fn=value_fn)
                ])
                visualizations[f'intent{idx}'] = image
            
            # GC value visualizations
            image_zz = viz_library.gcvalue_image(
                viz_env, viz_dataset, 
                functools.partial(get_v_zz, agent)
            )
            image_gz = viz_library.gcvalue_image(
                viz_env, viz_dataset,
                functools.partial(get_v_gz, agent, init_state)
            )
            visualizations['v_zz'] = image_zz
            visualizations['v_gz'] = image_gz

        for k, v in visualizations.items():
            visualizations[k] = torch.tensor(v, dtype=torch.uint8).permute(2, 0, 1)  # Convert to CHW format
        return visualizations

def get_values(agent, observations, intent):
    """Compute values for given observations and intent"""
    if not isinstance(observations, torch.Tensor):
        observations = torch.tensor(observations, dtype=torch.float32)
    intent_tiled = intent.repeat(observations.size(0), 1)
    v1, v2 = agent.value_net(observations, intent_tiled, intent_tiled)
    return (v1 + v2) / 2

def get_policy(agent, observations, intent):
    """Compute policy for given observations and intent"""
    def value_fn(obs):
        return get_values(agent, obs, intent).mean()
    
    # Compute gradients to get policy
    if not isinstance(observations, torch.Tensor):
        observations = torch.tensor(observations, dtype=torch.float32)
    observations.requires_grad_(True)
    value = value_fn(observations)
    value.backward()
    policy = observations.grad[:, :2]
    policy = policy / torch.norm(policy, dim=-1, keepdim=True)
    return policy.detach()

def get_debug_statistics(agent, batch):
    """Compute various debug statistics"""
    s = batch['observations']
    g = batch['goals']
    z = batch['desired_goals']
    
    # Compute different value combinations
    info_ssz = agent.value_net.get_info(s, s, z)
    info_szz = agent.value_net.get_info(s, z, z)
    info_sgz = agent.value_net.get_info(s, g, z)
    info_sgg = agent.value_net.get_info(s, g, g)
    info_szg = agent.value_net.get_info(s, z, g)
    
    stats = {}
    if 'phi' in info_sgz:
        stats['phi_norm'] = torch.norm(info_sgz['phi'], dim=-1).mean().item()
        stats['psi_norm'] = torch.norm(info_sgz['psi'], dim=-1).mean().item()
    
    stats.update({
        'v_ssz': info_ssz['v'].mean().item(),
        'v_szz': info_szz['v'].mean().item(),
        'v_sgz': info_sgz['v'].mean().item(),
        'v_sgg': info_sgg['v'].mean().item(),
        'v_szg': info_szg['v'].mean().item(),
        'diff_szz_szg': (info_szz['v'] - info_szg['v']).mean().item(),
        'diff_sgg_sgz': (info_sgg['v'] - info_sgz['v']).mean().item(),
    })
    return stats

def get_gcvalue(agent, s, g, z):
    """Get goal-conditioned value"""
    v_sgz_1, v_sgz_2 = agent.value_net(s, g, z)
    return (v_sgz_1 + v_sgz_2) / 2

def get_v_zz(agent, goal, observations):
    """Get value for goal-conditioned setting"""
    if not isinstance(observations, torch.Tensor):
        observations = torch.tensor(observations, dtype=torch.float32)
    if not isinstance(goal, torch.Tensor):
        goal = torch.tensor(goal, dtype=torch.float32)
    goal_tiled = goal.reshape((-1, goal.shape[-1])).repeat(observations.size(0), 1)
    return get_gcvalue(agent, observations, goal_tiled, goal_tiled)

def get_v_gz(agent, initial_state, target_goal, observations):
    """Get value for goal-conditioned setting with initial state"""
    if not isinstance(observations, torch.Tensor):
        observations = torch.tensor(observations, dtype=torch.float32)
    if not isinstance(initial_state, torch.Tensor):
        initial_state = torch.tensor(initial_state, dtype=torch.float32)
    if not isinstance(target_goal, torch.Tensor):
        target_goal = torch.tensor(target_goal, dtype=torch.float32)
    initial_state_tiled = initial_state.reshape((-1, initial_state.shape[-1])).repeat(observations.size(0), 1)
    target_goal_tiled = target_goal.reshape((-1, target_goal.shape[-1])).repeat(observations.size(0), 1)
    return get_gcvalue(agent, initial_state_tiled, observations, target_goal_tiled)

def get_traj_v(agent, trajectory):
    """Compute values along a trajectory"""
    observations = trajectory['observations']
    n = observations.size(0)
    
    # Compute all pairwise values
    all_values = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            s = observations[i].unsqueeze(0)
            g = observations[j].unsqueeze(0)
            all_values[i, j] = agent.value_net(s, g, g).mean().item()
    
    return {
        'dist_to_beginning': all_values[:, 0],
        'dist_to_end': all_values[:, -1],
        'dist_to_middle': all_values[:, n // 2],
    }

def parse_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='ICVF Training')
    parser.add_argument('--env_name', type=str, default='antmaze-large-diverse-v2', 
                        help='Environment name.')
    parser.add_argument('--save_dir', type=str, default='logdir/', 
                        help='Logging directory.')
    parser.add_argument('--seed', type=int, default=np.random.randint(1000000), 
                        help='Random seed.')
    parser.add_argument('--log_interval', type=int, default=1000, 
                        help='Metric logging interval.')
    parser.add_argument('--eval_interval', type=int, default=25000, 
                        help='Visualization interval.')
    parser.add_argument('--save_interval', type=int, default=100000, 
                        help='Save interval.')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Mini batch size.')
    parser.add_argument('--max_steps', type=int, default=int(1e6), 
                        help='Number of training steps.')
    parser.add_argument('--icvf_type', type=str, default='multilinear', 
                        choices=list(ICVF_REGISTRY.keys()), help='Which model to use.')
    parser.add_argument('--hidden_dims', type=str, default='256,256', 
                        help='Hidden sizes as comma-separated list.')
    parser.add_argument('--discount', type=float, default=0.99, 
                        help='Discount factor.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, 
                        help='Learning rate.')
    parser.add_argument('--eps', type=float, default=1e-8, 
                        help='Epsilon for Adam optimizer.')
    parser.add_argument('--experiment_id', type=str, default='exp1', 
                        help='Experiment identifier for TensorBoard.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create TensorBoard logger
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, args.experiment_id))
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, args.experiment_id)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    
    # Create environment and dataset
    env = d4rl_utils.make_env(args.env_name)
    dataset = d4rl_utils.get_dataset(env)
    gc_dataset = GCSDataset(dataset.data, **GCSDataset.get_default_config())
    example_batch = gc_dataset.sample(1)
    
    # Parse hidden dimensions
    hidden_dims = tuple(map(int, args.hidden_dims.split(',')))
    
    # Create ICVF model
    value_net = create_icvf(args.icvf_type, input_dim=example_batch['observations'].shape[-1], hidden_dims=hidden_dims)
    target_value_net = create_icvf(args.icvf_type, input_dim=example_batch['observations'].shape[-1], hidden_dims=hidden_dims)
    
    # Create learner
    agent = learner.create_learner(
        seed=args.seed,
        # observations=example_batch['observations'],
        value_net=value_net,
        target_value_net=target_value_net,
        discount=args.discount, # 0.99
        target_update_rate=0.005,
        expectile=0.9,
        optim_kwargs={'lr': args.learning_rate, # 3e-4
                      'eps': args.eps, # 1e-8
                      },
        no_intent=False,
        min_q=True,
        periodic_target_update=False,
    )
    
    # Create visualizer
    visualizer = DebugPlotGenerator(args.env_name, gc_dataset)
    
    # Training loop
    for step in tqdm(range(1, args.max_steps + 1)):
        batch = gc_dataset.sample(args.batch_size)
        update_info = agent.update(batch)
        
        # Log training metrics
        if step % args.log_interval == 0:
            debug_statistics = get_debug_statistics(agent, batch)
            for k, v in update_info.items():
                writer.add_scalar(f'training/{k}', v.item(), step)
            for k, v in debug_statistics.items():
                writer.add_scalar(f'pretraining/debug/{k}', v, step)
        
        # Log visualizations
        if step % args.eval_interval == 0:
            visualizations = visualizer.generate_debug_plots(agent)
            for k, v in visualizations.items():
                if isinstance(v, torch.Tensor):
                    writer.add_image(f'visualizations/{k}', v, step)
                else:
                    writer.add_scalar(f'visualizations/{k}', v, step)
        
        # Save model
        if step % args.save_interval == 0:
            save_path = os.path.join(save_dir, f'params_step_{step}.pth')
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'config': vars(args),
                'step': step
            }, save_path)
            print(f'Saved model to {save_path}')


if __name__ == '__main__':
    main()