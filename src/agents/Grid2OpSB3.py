# Copyright (c) 2020-2022 RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import inspect
import logging

from src.agents.grid2OpGymAgent import Grid2OpGymAgent

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import os
from typing import Optional


class SB3AgentGrid2Op(Grid2OpGymAgent):
    """This class represents the Agent (directly usable with grid2op framework)

    This agents uses the stable-baselines3 `nn_type` (by default PPO) as
    the neural network to take decisions on the grid.

    To be built, it requires:

    - `g2op_action_space`: a grid2op action space (used for initializing the grid2op agent)
    - `gym_act_space`: a gym observation space (used for the neural networks)
    - `gym_obs_space`: a gym action space (used for the neural networks)

    It can also accept different types of parameters:

    - `nn_type`: the type of "neural network" from stable baselines (by default PPO)
    - `nn_path`: the path where the neural network can be loaded from
    - `nn_kwargs`: the parameters used to build the neural network from scratch.

    Exactly one of `nn_path` and `nn_kwargs` should be provided. No more, no less.

    TODO heuristic part !

    Examples
    ---------

    The best way to have such an agent is either to train it:

    .. code-block:: python

        from l2rpn_baselnes.PPO_SB3 import train
        agent = train(...)  # see the doc of the `train` function !

    Or you can also load it when you evaluate it (after it has been trained !):

    .. code-block:: python

        from l2rpn_baselnes.PPO_SB3 import evaluate
        agent = evaluate(...)  # see the doc of the `evaluate` function !

    To create such an agent from scratch (NOT RECOMMENDED), you can do:

    .. code-block:: python

        import grid2op
        from src.envs.gym_compat import BoxGymObsSpace, BoxGymActSpace, GymEnv
        from lightsim2grid import LightSimBackend

        from l2rpn_baselnes.PPO_SB3 import PPO_SB3

        env_name = "l2rpn_case14_sandbox"  # or any other name

        # customize the observation / action you want to keep
        obs_attr_to_keep = ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p", "load_q",
                            "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                            "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                            "storage_power", "storage_charge"]
        act_attr_to_keep = ["redispatch", "curtail", "set_storage"]

        # create the grid2op environment
        env = grid2op.make(env_name, backend=LightSimBackend())

        # define the action space and observation space that your agent
        # will be able to use
        env_gym = GymEnv(env)
        env_gym.observation_space.close()
        env_gym.observation_space = BoxGymObsSpace(env.observation_space,
                                                attr_to_keep=obs_attr_to_keep)
        env_gym.action_space.close()
        env_gym.action_space = BoxGymActSpace(env.action_space,
                                            attr_to_keep=act_attr_to_keep)

        # create the key word arguments used for the NN
        nn_kwargs = {
            "policy": MlpPolicy,
            "env": env_gym,
            "verbose": 0,
            "learning_rate": 1e-3,
            "tensorboard_log": ...,
            "policy_kwargs": {
                "net_arch": [100, 100, 100]
            }
        }

        # create a grid2gop agent based on that (this will reload the save weights)
        grid2op_agent = PPO_SB3(env.action_space,
                                env_gym.action_space,
                                env_gym.observation_space,
                                nn_kwargs=nn_kwargs  # don't load it from anywhere
                               )

    """

    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 nn_type,
                 nn_path=None,
                 nn_kwargs=None,
                 custom_load_dict=None,
                 gymenv=None,
                 eval_env = None,
                 iter_num=None,
                 ):
        self._nn_type = nn_type
        if custom_load_dict is not None:
            self.custom_load_dict = custom_load_dict
        else:
            self.custom_load_dict = {}
        self.eval_env = eval_env
        self._iter_num: Optional[int] = iter_num
        super().__init__(g2op_action_space, gym_act_space, gym_obs_space,
                         nn_path=nn_path, nn_kwargs=nn_kwargs,
                         gymenv=gymenv
                         )

    def get_act(self, gym_obs, reward, done):
        """Retrieve the gym action from the gym observation and the reward.
        It only (for now) work for non recurrent policy.

        Parameters
        ----------
        gym_obs : gym observation
            The gym observation
        reward : ``float``
            the current reward
        done : ``bool``
            whether the episode is over or not.

        Returns
        -------
        gym action
            The gym action, that is processed in the :func:`GymAgent.act`
            to be used with grid2op
        """
        action, _ = self.nn_model.predict(gym_obs, deterministic=False)
        return action

    def load(self):
        """
        Load the NN model.

        In the case of a PPO agent, this is equivalent to perform the:

        .. code-block:: python

            PPO.load(nn_path)
        """
        custom_objects = {"action_space": self._gym_act_space,
                          "observation_space": self._gym_obs_space}
        for key, val in self.custom_load_dict.items():
            custom_objects[key] = val
        path_load = self._nn_path
        if self._iter_num is not None:
            path_load = path_load + f"_{self._iter_num}_steps"
        log.debug(F"loading agent from [{path_load}]")
        self.nn_model = self._nn_type.load(path_load,
                                           custom_objects=custom_objects,
                                           env=self.gymenv)

    def build(self):
        """Create the underlying NN model from scratch.

        In the case of a PPO agent, this is equivalent to perform the:

        .. code-block:: python

            PPO(**nn_kwargs)
        """
        self.nn_model = self._nn_type(**self._nn_kwargs)

    def learn(self,
          total_timesteps=1,
          save_path=None,
          callbacks={},
          **learn_kwargs):

        list_of_callbacks=[]
        for callback_name,callback_data in callbacks.items():
            callback_class=callback_data.get("class")
            if inspect.isclass(callback_class) or inspect.isfunction(callback_class):
                kwargs = callback_data.get("kwargs",{})
                callback = callback_class(**kwargs)
            else:
                callback= callback_class
            log.info(F"Configured callback [{callback_name}] using class [{callback_class}]")
            list_of_callbacks.append(callback)
        if learn_kwargs is None:
            learn_kwargs={}
        # train it
        self.nn_model.learn(total_timesteps=total_timesteps,
                             callback=list_of_callbacks,
                             eval_env=self.eval_env,
                            **learn_kwargs
                             )

        # save it
        if save_path is not None:
            self.nn_model.save(save_path)

