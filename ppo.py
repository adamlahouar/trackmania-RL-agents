# TMRL imports
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.util import partial, prod, cached_property
from tmrl.actor import TorchActorModule
from tmrl.networking import Trainer, RolloutWorker, Server
from tmrl.training_offline import TrainingOffline
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad

# Torch imports
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch import Tensor

# Other imports
import os
import numpy as np
from math import floor
import json
from copy import deepcopy
import itertools
from argparse import ArgumentParser
import math


class ConfigParameters:
    def __init__(self):
        # Useful parameters
        self.epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
        self.rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
        self.steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
        self.start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]
        self.max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
        self.update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
        self.update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]
        self.device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'
        self.memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
        self.batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

        # Wandb parameters
        self.wandb_run_id = cfg.WANDB_RUN_ID
        self.wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]
        self.wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]
        self.wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]
        os.environ['WANDB_API_KEY'] = self.wandb_key

        self.max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

        # Networking parameters
        self.server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER
        self.server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER
        self.server_port = cfg.PORT
        self.password = cfg.PASSWORD
        self.security = cfg.SECURITY

        # Advanced parameters
        self.memory_base_cls = cfg_obj.MEM
        self.sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
        self.sample_preprocessor = None
        self.dataset_path = cfg.DATASET_PATH
        self.obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

        # Competition fixed parameters (we don't care)
        self.env_cls = cfg_obj.ENV_CLS
        self.device_worker = 'cuda' if cfg.CUDA_INFERENCE else 'cpu'

        # Environment parameters
        self.window_width = cfg.WINDOW_WIDTH
        self.window_height = cfg.WINDOW_HEIGHT
        self.img_width = cfg.IMG_WIDTH
        self.img_height = cfg.IMG_HEIGHT
        self.img_grayscale = cfg.GRAYSCALE
        self.imgs_buf_len = cfg.IMG_HIST_LEN
        self.act_buf_len = cfg.ACT_BUF_LEN


params = ConfigParameters()

# Memory class
memory_cls = partial(params.memory_base_cls,
                     memory_size=params.memory_size,
                     batch_size=params.batch_size,
                     sample_preprocessor=params.sample_preprocessor,
                     dataset_path=params.dataset_path,
                     imgs_obs=params.imgs_buf_len,
                     act_buf_len=params.act_buf_len,
                     crc_debug=False)


class TorchJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


class PPOActorModule(TorchActorModule):

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        obs_dim = sum(prod(s for s in space.shape) for space in observation_space)
        act_dim = action_space.shape[0]

        self.network = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 256))

        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)

    def save(self, path):
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)

    def load(self, path, device):
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        return self

    def _process_observation(self, observation: tuple[Tensor]) -> Tensor:
        processed_observations = []

        for tensor in observation:
            tensor = tensor.flatten(start_dim=1)
            processed_observations.append(tensor)

        return torch.cat(processed_observations, -1)

    def forward(self, observation: tuple[Tensor], test=False, compute_logprob=True) -> tuple[Tensor, any]:
        processed_observation = self._process_observation(observation)
        network_output = self.network(processed_observation)

        mu = self.mu_layer(network_output)

        log_std = self.log_std_layer(network_output)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)

        if test:
            action = mu
        else:
            action = pi_distribution.rsample()

        if compute_logprob:
            logprob = pi_distribution.log_prob(action).sum(axis=-1)
            logprob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        else:
            logprob = None

        action = torch.tanh(action)
        action = action.squeeze()
        return action, logprob

    def act(self, observation: tuple[Tensor], test=False) -> Tensor:
        with torch.no_grad():
            action = self.forward(observation, test, compute_logprob=False)[0].flatten()
            return action.cpu().numpy()


class PPOCriticModule(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(83, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 1))

    def _process_observation(self, observation: tuple[Tensor]) -> Tensor:
        processed_observations = []

        for tensor in observation:
            tensor = tensor.flatten(start_dim=1)
            processed_observations.append(tensor)

        return torch.cat(processed_observations, -1)

    def forward(self, observation: tuple[Tensor]):
        processed_observation = self._process_observation(observation)
        q = self.network(processed_observation)
        return torch.squeeze(q, -1)


class VanillaCNNPPO(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.actor = PPOActorModule(observation_space, action_space)
        self.q1 = PPOCriticModule(observation_space, action_space)

        # TODO: remove q2?
        # self.q2 = PPOCriticModule(observation_space, action_space)


class PPOTrainingAgent(TrainingAgent):
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=VanillaCNNPPO,
                 gamma=0.99,
                 polyak=0.995,
                 alpha=0.2,
                 lr_actor=1e-3,
                 lr_critic=1e-3):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        model = model_cls(observation_space, action_space)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        # self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())

        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.model.q1.parameters(), lr=self.lr_critic)
        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

        # TODO: put these in the config
        self.lambda_ = 0.95
        self.clip_ratio = 0.2

    def get_actor(self):
        return self.model_nograd.actor

    def _compute_advantages_and_returns(self, rewards, values, terminates):

        trajectory_length = rewards.size(0)

        deltas = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        previous_return = 0
        previous_value = 0
        previous_advantage = 0

        for t in reversed(range(trajectory_length)):
            if t == trajectory_length - 1:
                next_non_terminal = 1.0 - terminates[-1]
                next_value = values[-1]
            else:
                next_non_terminal = 1.0 - terminates[t]
                next_value = values[t + 1]

            deltas[t] = rewards[t] + self.gamma * next_non_terminal * next_value - values[t]
            returns[t] = deltas[t] + self.gamma * self.lambda_ * next_non_terminal * previous_return
            advantages[t] = deltas[t] + self.gamma * self.lambda_ * next_non_terminal * previous_advantage

            previous_return = returns[t]
            previous_advantage = advantages[t]

        return advantages, returns

    def _compute_loss_pi(self, observations, actions, advantages, logp_old):

        pi, logp = self.model.actor(observations, test=False, compute_logprob=True)

        ratio = torch.exp(logp - logp_old)

        clipped_advantages = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        loss_pi = -torch.min(ratio * advantages, clipped_advantages).mean()

        # kl = (logp_old - logp).mean().item()
        # ent = pi.entropy().mean().item()

        # clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        # clipfrac = torch.as_tensor(clipped, dtype=torch.float).mean().item()

        # pi_info = {'kl': kl, 'ent': ent, 'cf': clipfrac}

        return loss_pi  # , pi_info

    def _update_target_model(self):
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * param.data)

    def train(self, batch):

        # Decompose batch into relevant components
        initial_observations, actions, rewards, final_observations, terminates, _ = batch

        # Get value estimates from critic
        values = self.model.q1(initial_observations)

        # Calculate advantages and returns
        advantages, returns = self._compute_advantages_and_returns(rewards, values, terminates)

        # Get old log probabilities from actor
        _, logp_old = self.model.actor(initial_observations, test=False, compute_logprob=True)

        # Compute actor and critic losses
        loss_actor = self._compute_loss_pi(initial_observations, actions, advantages, logp_old)
        loss_critic = ((values - returns) ** 2).mean()

        # TODO: figure out why this crashes

        # Take optimization step for actor
        self.pi_optimizer.zero_grad()
        loss_actor.backward()
        self.pi_optimizer.step()

        # Take optimization step for critic
        self.q_optimizer.zero_grad()
        loss_critic.backward()
        self.q_optimizer.step()

        # Update target networks
        self._update_target_model()

        ret_dict = dict(
            loss_actor=loss_actor.detach().item(),
            loss_critic=loss_critic.detach().item()
        )

        return ret_dict


# TrainingAgent class
training_agent_cls = partial(PPOTrainingAgent,
                             model_cls=VanillaCNNPPO,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.02,
                             lr_actor=5e-6,
                             lr_critic=3e-5)

# Training class
training_cls = partial(TrainingOffline,
                       env_cls=params.env_cls,
                       memory_cls=memory_cls,
                       training_agent_cls=training_agent_cls,
                       epochs=params.epochs,
                       rounds=params.rounds,
                       steps=params.steps,
                       update_buffer_interval=params.update_buffer_interval,
                       update_model_interval=params.update_model_interval,
                       max_training_steps_per_env_step=params.max_training_steps_per_env_step,
                       start_training=params.start_training,
                       device=params.device_trainer)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    if args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=params.server_ip_for_trainer,
                             server_port=params.server_port,
                             password=params.password,
                             security=params.security)
        my_trainer.run()
    elif args.worker or args.test:
        rw = RolloutWorker(env_cls=params.env_cls,
                           actor_module_cls=PPOActorModule,
                           sample_compressor=params.sample_compressor,
                           device=params.device_worker,
                           server_ip=params.server_ip_for_worker,
                           server_port=params.server_port,
                           password=params.password,
                           security=params.security,
                           max_samples_per_episode=params.max_samples_per_episode,
                           obs_preprocessor=params.obs_preprocessor,
                           standalone=args.test)
        rw.run()
    elif args.server:
        import time

        serv = Server(port=params.server_port,
                      password=params.password,
                      security=params.security)
        while True:
            time.sleep(1.0)
