# TMRL imports
import json
# Other imports
import os
import random
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
# Torch imports
import torch
import torch.nn as nn
from tmrl.actor import TorchActorModule
from tmrl.custom.custom_memories import MemoryTM, last_true_in_list, replace_hist_before_eoe
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.memory import check_samples_crc
from tmrl.networking import Trainer, RolloutWorker, Server
from tmrl.training import TrainingAgent
from tmrl.training_offline import TrainingOffline
from tmrl.util import partial, prod, cached_property
from torch import Tensor
from torch.distributions.normal import Normal
from torch.optim import Adam


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

        # Custom stuff
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.lambda_ = 0.97
        self.clip_ratio = 0.2
        self.max_grad_norm = 0.5
        self.training_iterations = 80
        self.target_kl = 0.01


params = ConfigParameters()


class PPOCustomMemory(MemoryTM):

    def get_transition(self, item):
        """
        CAUTION: item is the first index of the 4 images in the images history of the OLD observation
        CAUTION: in the buffer, a sample is (act, obs(act)) and NOT (obs, act(obs))
            i.e. in a sample, the observation is what step returned after being fed act (and preprocessed)
            therefore, in the RTRL setting, act is appended to obs
        So we load 5 images from here...
        Don't forget the info dict for CRC debugging
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:  # if first item of the buffer
                item += 1
            elif item == self.__len__() - 1:  # if last item of the buffer
                item -= 1
            elif random.random() < 0.5:  # otherwise, sample randomly
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # if a reset transition has influenced the observation, special care must be taken
        last_eoes = self.data[4][idx_now - self.min_samples:idx_now]  # self.min_samples values
        last_eoe_idx = last_true_in_list(last_eoes)  # last occurrence of True

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1)
            replace_hist_before_eoe(hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset)
            replace_hist_before_eoe(hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1)
            replace_hist_before_eoe(hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset)

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[7][idx_now]
        truncated = self.data[8][idx_now]
        info = self.data[6][idx_now]
        logprobs = self.data[9][idx_now]
        rtgs = self.data[10][idx_now]

        return last_obs, new_act, rew, new_obs, terminated, truncated, info, logprobs, rtgs

    def load_imgs(self, item):
        res = self.data[3][(item + self.start_imgs_offset):(item + self.start_imgs_offset + self.imgs_obs + 1)]
        return np.stack(res)

    def load_acts(self, item):
        res = self.data[1][(item + self.start_acts_offset):(item + self.start_acts_offset + self.act_buf_len + 1)]
        return res

    def append_buffer(self, buffer):
        """
        buffer is a list of samples (act, obs, rew, terminated, truncated, info)
        don't forget to keep the info dictionary in the sample for CRC debugging
        """

        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # lidar
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes (terminated or truncated)
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[3] for b in buffer.memory]  # terminated
        d8 = [b[4] for b in buffer.memory]  # truncated
        d9 = [b[6] for b in buffer.memory]  # old logprobs
        d10 = [b[7] for b in buffer.memory]  # rtgs

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7
            self.data[8] += d8
            self.data[9] += d9
            self.data[10] += d10
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)
            self.data.append(d7)
            self.data.append(d8)
            self.data.append(d9)
            self.data.append(d10)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            self.data[0] = self.data[0][to_trim:]
            self.data[1] = self.data[1][to_trim:]
            self.data[2] = self.data[2][to_trim:]
            self.data[3] = self.data[3][to_trim:]
            self.data[4] = self.data[4][to_trim:]
            self.data[5] = self.data[5][to_trim:]
            self.data[6] = self.data[6][to_trim:]
            self.data[7] = self.data[7][to_trim:]
            self.data[8] = self.data[8][to_trim:]
            self.data[9] = self.data[9][to_trim:]
            self.data[10] = self.data[10][to_trim:]

        return self

    def __getitem__(self, item):
        prev_obs, new_act, rew, new_obs, terminated, truncated, info, logprobs, rtgs = self.get_transition(item)
        if self.crc_debug:
            po, a, o, r, d, t = info['crc_sample']
            debug_ts, debug_ts_res = info['crc_sample_ts']
            check_samples_crc(po, a, o, r, d, t, prev_obs, new_act, new_obs, rew, terminated, truncated, debug_ts,
                              debug_ts_res)
        if self.sample_preprocessor is not None:
            prev_obs, new_act, rew, new_obs, terminated, truncated = self.sample_preprocessor(prev_obs, new_act, rew,
                                                                                              new_obs, terminated,
                                                                                              truncated)
        terminated = np.float32(terminated)  # we don't want bool tensors
        truncated = np.float32(truncated)  # we don't want bool tensors
        return prev_obs, new_act, rew, new_obs, terminated, truncated, logprobs, rtgs


# Memory class
memory_cls = partial(PPOCustomMemory,  # params.memory_base_cls
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


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class PPOActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.Tanh):
        super().__init__(observation_space, action_space)
        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False

        dim_act = action_space.shape[0]
        log_std = -0.5 * np.ones(dim_act, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_layer = mlp([dim_obs] + list(hidden_sizes) + [dim_act], activation)

        self.act_limit = action_space.high[0]

    def _distribution(self, obs):
        mu = self.mu_layer(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        act = torch.as_tensor(act)
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        obs = torch.cat(obs, -1)
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        a = pi.sample()
        a = torch.tanh(a)
        a = self.act_limit * a
        res = a.squeeze().cpu().numpy()
        if not len(res.shape):
            res = np.expand_dims(res, 0)

        return res, logp_a

    def act(self, obs: tuple[Tensor], test=False):
        obs = torch.cat(obs, -1)
        with torch.no_grad():
            pi = self._distribution(obs)
            a = pi.sample()
            logp_a = self._log_prob_from_distribution(pi, a)

        a = torch.tanh(a)
        a = self.act_limit * a
        res = a.squeeze().cpu().numpy()
        if not len(res.shape):
            res = np.expand_dims(res, 0)

        return res


class PPOCriticModule(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.Tanh):
        super().__init__()
        try:
            obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(obs_space.shape)
            self.tuple_obs = False

        self.q = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        q = self.q(x)
        return torch.squeeze(q, -1)


class PPOActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        self.actor = PPOActorModule(observation_space, action_space)
        self.critic = PPOCriticModule(observation_space, action_space)


class PPOTrainingAgent(TrainingAgent):
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=PPOActorCritic,
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

        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.model.critic.parameters(), lr=self.lr_critic)
        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

        self.lambda_ = params.lambda_
        self.clip_ratio = params.clip_ratio
        self.max_grad_norm = params.max_grad_norm
        self.training_iterations = params.training_iterations

    def get_actor(self):
        return self.model_nograd.actor

    def _compute_loss_pi(self, observations, actions, advantages, logp_old):
        _, logp = self.model.actor(observations, actions)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        loss_pi = -(torch.min(ratio * advantages, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return loss_pi, approx_kl, clipfrac

    def _compute_loss_v(self, observation, returns):
        return ((self.model.critic(observation) - returns) ** 2).mean()

    def _compute_values_advantages(self, initial_observations, final_observations, rewards, terminates, truncates):
        values = self.model.critic(initial_observations)
        final_values = self.model.critic(final_observations)

        advantages = torch.zeros_like(rewards)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.gamma * final_values[t] * (1 - terminates[t]) - values[t]
                advantages[t] = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - terminates[t]) - values[t]
                advantages[t] = delta + self.gamma * self.lambda_ * (1 - truncates[t]) * advantages[t + 1]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        return values, advantages

    def train(self, batch):
        initial_observations, actions, rewards, final_observations, terminates, truncates, logp_old, rtgs = batch
        values, advantages = self._compute_values_advantages(initial_observations, final_observations, rewards,
                                                             terminates, truncates)

        pi_l_old, approx_kl, clipfrac = self._compute_loss_pi(initial_observations, actions, advantages, logp_old)
        pi_l_old = pi_l_old.item()
        v_l_old = self._compute_loss_v(initial_observations, rtgs).item()

        for i in range(self.training_iterations):
            self.pi_optimizer.zero_grad()
            loss_pi, _, _ = self._compute_loss_pi(initial_observations, actions, advantages.detach(), logp_old)
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
            self.pi_optimizer.step()

        for i in range(self.training_iterations):
            self.q_optimizer.zero_grad()
            loss_v = self._compute_loss_v(initial_observations, rtgs)
            loss_v.backward()
            self.q_optimizer.step()

        ret_dict = dict(
            loss_actor=pi_l_old,
            loss_critic=v_l_old,
            approx_kl=approx_kl,
            clipfrac=clipfrac
        )

        return ret_dict


# TrainingAgent class
training_agent_cls = partial(PPOTrainingAgent,
                             model_cls=PPOActorCritic,
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
