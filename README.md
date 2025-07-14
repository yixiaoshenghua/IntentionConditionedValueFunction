# ICVF (Reinforcement Learning from Passive Data via Latent Intentions)

This repository contains PyTorch code for the paper [Reinforcement Learning from Passive Data via Latent Intentions](https://arxiv.org/abs/2304.04782).. 

## Installation

Add this directory to your PYTHONPATH. Install the dependencies for ICVF, and additional dependencies depending on which environments you want to try (see requirements.txt).

The XMagical dataset is available on [Google Drive](https://drive.google.com/drive/folders/1qDiOoKrWUybJBB4dIzz6-lWy7Z1MAYro?usp=sharing)


### Examples

To train an ICVF agent on the Antmaze dataset, run:

```
python experiments/antmaze/pretrain_icvf.py --env_name=antmaze-large-diverse-v2 --seed 1 
--log_interval 500 --eval_interval 10000 --save_interval 100000 --experiment_id antmaze-large-diverse-v2/seed1
```


To train an ICVF agent on the XMagical dataset, run:

```
python experiments/xmagical/train_icvf.py
```


### Code Structure:

- [rl_utils/](rl_utils/): A helper library for rl algorithms
- [icvf_envs/](icvf_envs/): Environment wrappers and dataset loaders
- [src/](src/): New code for ICVF
- [icvf_learner.py](src/icvf_learner.py): Core algorithmic logic
- [icvf_networks.py](src/icvf_networks.py): ICVF network architecture
- [extra_agents/](src/extra_agents/): Finetuning downstream RL agents from the ICVF representation
- [experiments/](experiments/): Launchers for ICVF experiments

