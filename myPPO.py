from stable_baselines3 import PPO

# ALL IMPORTS OF ppo.py
import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

# from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

SelfPPO = TypeVar("SelfPPO", bound="PPO")


# ALL IMPORTS OF on_policy_algorithm.py
import sys
import time
# from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# import numpy as np
# import torch as th
# from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

class MyPPO(PPO):

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
        ) -> bool:
            """
            Collect experiences using the current policy and fill a ``RolloutBuffer``.
            The term rollout here refers to the model-free notion and should not
            be used with the concept of rollout used in model-based RL or planning.

            :param env: The training environment
            :param callback: Callback that will be called at each step
                (and at the beginning and end of the rollout)
            :param rollout_buffer: Buffer to fill with rollouts
            :param n_rollout_steps: Number of experiences to collect per environment
            :return: True if function returned with at least `n_rollout_steps`
                collected, False if callback terminated rollout prematurely.
            """
            assert self._last_obs is not None, "No previous observation was provided"
            # Switch to eval mode (this affects batch norm / dropout)
            self.policy.set_training_mode(False)

            n_steps = 0
            rollout_buffer.reset()
            # Sample new weights for the state dependent exploration
            if self.use_sde:
                self.policy.reset_noise(env.num_envs)

            callback.on_rollout_start()

            #FIX "dones" potrebbe non essere associato
            dones = np.zeros(n_rollout_steps, dtype=bool)

            while n_steps < n_rollout_steps:
                if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.policy.reset_noise(env.num_envs)

                with th.no_grad():
                    # Convert to pytorch tensor or to TensorDict
                    obs_tensor = obs_as_tensor(self._last_obs, self.device) # type: ignore
                    actions, values, log_probs = self.policy(obs_tensor)
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions

                if isinstance(self.action_space, spaces.Box):
                    if self.policy.squash_output:
                        # Unscale the actions to match env bounds
                        # if they were previously squashed (scaled in [-1, 1])
                        clipped_actions = self.policy.unscale_action(clipped_actions)
                    else:
                        # Otherwise, clip the actions to avoid out of bound error
                        # as we are sampling from an unbounded Gaussian distribution
                        clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                new_obs, rewards, dones, infos = env.step(clipped_actions)

                self.num_timesteps += env.num_envs

                # Give access to local variables
                callback.update_locals(locals())
                if not callback.on_step():
                    return False

                self._update_info_buffer(infos)
                n_steps += 1

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                # Handle timeout by bootstraping with value function
                # see GitHub issue #633
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                        rewards[idx] += self.gamma * terminal_value

                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )
                self._last_obs = new_obs  # type: ignore[assignment]
                self._last_episode_starts = dones

            with th.no_grad():
                # Compute value for the last timestep
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

            rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

            callback.update_locals(locals())

            callback.on_rollout_end()

            return True