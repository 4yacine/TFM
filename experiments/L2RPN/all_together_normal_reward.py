import sys
import os
from pathlib import Path

from grid2op.Reward import EpisodeDurationReward, CloseToOverflowReward

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.absolute())
sys.path.append(AutoGridPath)

from src.helpers import create_experiment_gitignore


import grid2op
from src import constants, AutoGrid
from stable_baselines3 import DDPG, SAC, TD3, DQN, PPO, A2C
from stable_baselines3.a2c import MlpPolicy as a2cMlpPolicy
from stable_baselines3.ddpg import MlpPolicy as ddpgMlpPolicy
from stable_baselines3.sac import MlpPolicy as sacMlpPolicy
from stable_baselines3.td3 import MlpPolicy as td3MlpPolicy
from stable_baselines3.dqn import MlpPolicy as dqnMlpPolicy
from stable_baselines3.ppo import MlpPolicy as ppoMlpPolicy
from src.makers.SB3 import create_agent_sb3
from src.envs.gymenv_heuristics import GymEnvWithHeuristics
from grid2op.gym_compat import GymEnv, BoxGymActSpace, BoxGymObsSpace, DiscreteActSpace
from typing import List
from grid2op.Action import BaseAction
import numpy as np
from lightsim2grid import LightSimBackend
import logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
class CustomGymEnv(GymEnvWithHeuristics):
    """This environment is slightly more complex that the other one.

    It consists in 2 things:

    #. reconnecting the powerlines if possible
    #. doing nothing is the state of the grid is "safe" (for this class, the notion of "safety" is pretty simple: if all
        flows are bellow 90% (by default) of the thermal limit, then it is safe)

    If for a given step, non of these things is applicable, the underlying trained agent is asked to perform an action

    .. warning::
        When using this environment, we highly recommend to adapt the parameter `safe_max_rho` to suit your need.

        Sometimes, 90% of the thermal limit is too high, sometimes it is too low.

    """

    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, **kwargs):
        super().__init__(env_init, reward_cumul=reward_cumul, *args, **kwargs)
        self._safe_max_rho = safe_max_rho
        self.dn = self.init_env.action_space({})

    def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
        """To match the description of the environment, this heuristic will:

        - return the list of all the powerlines that can be reconnected if any
        - return the list "[do nothing]" is the grid is safe
        - return the empty list (signaling the agent should take control over the heuristics) otherwise

        Parameters
        ----------
        See parameters of :func:`GymEnvWithHeuristics.heuristic_actions`

        Returns
        -------
        See return values of :func:`GymEnvWithHeuristics.heuristic_actions`
        """

        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        res = []
        if np.any(to_reco):
            # reconnect something if it can be
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
                res.append(g2op_act)
        elif g2op_obs.rho.max() <= self._safe_max_rho:
            # play do nothing if there is "no problem" according to the "rule of thumb"
            res = [self.init_env.action_space()]
        return res

    def step(self, gym_action):
        """This function implements the special case of the "step" function (as seen by the "gym environment") that might
        call multiple times the "step" function of the underlying "grid2op environment" depending on the
        heuristic.

        It takes a gym action, convert it to a grid2op action (thanks to the action space) and
        simulates if this action is better than doing nothing. If so, it performs the action otherwise
        it performs the "do nothing" action.

        Then process the heuristics / expert rules / forced actions / etc. and return the next gym observation that will
        be processed by the agent.

        The number of "grid2op steps" can vary between different "gym environment" call to "step".

        It has the same signature as the `gym.Env` "step" function, of course.

        Parameters
        ----------
        gym_action :
            the action (represented as a gym one) that the agent wants to perform.

        Returns
        -------
        gym_obs:
            The gym observation that will be processed by the agent

        reward: ``float``
            The reward of the agent (that might be computed by the )

        done: ``bool``
            Whether the episode is over or not

        info: Dict
            Other type of informations

        """
        g2op_act = self.action_space.from_gym(gym_action)

        _, sim_reward_act, _, _ = self.init_env.simulate(g2op_act)
        _, sim_reward_dn, _, _ = self.init_env.simulate(self.dn)
        if sim_reward_dn > sim_reward_act:
            g2op_act = self.dn

        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        if not done:
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, float(reward), done, info


SAVE_PATH = "./agents"
config = {
    "core": {
        "logger": {
            "console": {
                "level": "INFO"
            },
            "file": {
                "level": "DEBUG",
                "filename": "all_execution_log.log",
                "mode": "w"
            }
        }
    },
    "common": {
        "save_path": SAVE_PATH,
        "env": {
            "simulator": grid2op,
            "env_class": grid2op.make,
            #"env_backend_class":LightSimBackend,
            "env_kwargs": {
                "dataset": "l2rpn_wcci_2022",
                "difficulty": "competition",
                "reward_class":CloseToOverflowReward,
                "backend":LightSimBackend
            },
            "gymenv_class": CustomGymEnv
        },
        "action_space": {
            "class":BoxGymActSpace,
            "action_space_kwargs":{
                "attr_to_keep":[
                    #"set_line_status",
                    #"change_line_status",
                    #"set_bus",
                    #"change_bus",
                    "redispatch",
                    "set_storage",
                    "curtail"
                ]
            }
        },
        "observation_space": {
            "class": BoxGymObsSpace,
            "observation_space_kwargs":{
                "attr_to_keep":[
                    "gen_p",
                    #"gen_p_before_curtail",
                    "gen_q",
                    "gen_v",
                    #"gen_margin_up",
                    #"gen_margin_down",
                    #"load_p",
                    #"load_q",
                    #"load_v",
                    "p_or",
                    "q_or",
                    "v_or",
                    "a_or",
                    "p_ex",
                    "q_ex",
                    "v_ex",
                    "a_ex",
                    "rho",
                    #"line_status",
                    #"timestep_overflow",
                    #"topo_vect",
                    #"time_before_cooldown_line",
                    #"time_before_cooldown_sub",
                    #"time_next_maintenance",
                    #"duration_next_maintenance",
                    #"target_dispatch",
                    #"actual_dispatch",
                    "storage_charge",
                    "storage_power_target",
                    #"storage_power",
                    "curtailment",
                    "curtailment_limit",
                    #"curtailment_limit_effective",
                    #"thermal_limit",
                    #"theta_or",
                    #"theta_ex",
                    "load_theta",
                    "gen_theta"]
                }
        },
        "agent": {
            "maker": create_agent_sb3,
            "net_kwargs":{
                "tensorboard_log": os.path.join(SAVE_PATH, "tensorboard_log"),
                "policy_kwargs": {
                    "net_arch": [256, 128, 64, 35],
                }
            }
        },
        "training": constants.TRAINING_DEFAULT,
        "training_kwargs": {
            "total_timesteps": 10000000,
            "save_path": SAVE_PATH
        },
        "evaluation":  constants.EVALUATION_L2RPN2022,
    },
    "experiments": {
         "experiment_A2C": {
             "name": "A2C",
             "agent": {
                 "class": A2C,
                 "net_kwargs":{
                     "n_steps":1,
                     "policy": a2cMlpPolicy
                 }
             },
             "evaluation_kwargs":{
                 "save_path":os.path.join(SAVE_PATH,"A2C")
             }
         },
        "experiment_DDPG": {
            "name": "DDPG",
            "agent": {
                "class": DDPG,
                "net_kwargs":{
                    "policy": ddpgMlpPolicy
                }
            },
            "evaluation_kwargs":{
                "save_path":os.path.join(SAVE_PATH,"DDPG")
            }
        },
        "experiment_SAC": {
            "name": "SAC",
            "agent": {
                "class": SAC,
                "net_kwargs":{
                    "policy": sacMlpPolicy
                }
            },
            "evaluation_kwargs":{
                "save_path":os.path.join(SAVE_PATH,"SAC")
            }
        },
        "experiment_TD3": {
            "name": "TD3",
            "agent": {
                "class": TD3,
                "net_kwargs":{
                    "policy": td3MlpPolicy
                }
            },
            "evaluation_kwargs":{
                "save_path":os.path.join(SAVE_PATH,"TD3")
            }
        },
        "experiment_DQN": {
            "name": "DQN",
            "action_space": {
                "class":DiscreteActSpace,
            },
            "agent": {
                "class": DQN,
                "net_kwargs":{
                    "policy": dqnMlpPolicy
                }
            },
            "evaluation_kwargs":{
                "save_path":os.path.join(SAVE_PATH,"DQN")
            }
        },
        "experiment_PPO": {
            "name": "PPO",
            "action_space": {
                "class":DiscreteActSpace,
            },
            "agent": {
                "class": PPO,
                "net_kwargs":{
                    "policy": ppoMlpPolicy
                }
            },
            "evaluation_kwargs":{
                "save_path":os.path.join(SAVE_PATH,"PPO")
            }
        }
    }
}


# Function to create the configuration for the main execution
# Its mandatory to have this function declared with this signature.
def get_config():
    return config


if __name__ == "__main__":
    main = AutoGrid.main(config,force_log="DEBUG")
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()
