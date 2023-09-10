import sys
import os
from pathlib import Path

# This is incase you execute this example with a clone repository, so python can find AutoGrid
AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


from src.helpers import create_experiment_gitignore
import numpy as np
from grid2op.Reward import EpisodeDurationReward
from lightsim2grid import LightSimBackend

from src.envs.gymenv_heuristics import GymEnvWithHeuristics

import grid2op
from grid2op.Converter import IdToAct

from src import constants, AutoGrid
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from src.makers.SB3 import create_agent_sb3


def _filter_action(action):
    MAX_ELEM = 2
    act_dict = action.impact_on_objects()
    # Remove Curtailment
    if act_dict["curtailment"]["changed"] or action._modif_curtailment or len(act_dict["curtailment"]["limit"]) > 0:
        return False
    elem = 0
    elem += act_dict["force_line"]["reconnections"]["count"]
    elem += act_dict["force_line"]["disconnections"]["count"]
    elem += act_dict["switch_line"]["count"]
    elem += len(act_dict["topology"]["bus_switch"])
    elem += len(act_dict["topology"]["assigned_bus"])
    elem += len(act_dict["topology"]["disconnect_bus"])
    elem += len(act_dict["redispatch"]["generators"])

    if elem <= MAX_ELEM:
        return True
    return False


def create_action_space(env_action_space, load_path=None, save_path=None):
    converter = IdToAct(env_action_space)
    if load_path is not None:
        try:
            my_path = os.path.join(load_path, AGENT_NAME)
            converter.init_converter(all_actions=os.path.join(my_path, "filtered_actions.npy"))
            log.debug(F"Loaded Action space size:{converter.n} from folder [{my_path}]")
            return DiscreteActSpace(converter, action_list=converter.all_actions)
        except FileNotFoundError as e:
            log.warning(F"Could not load filtered action space, one will be created now. Error: {str(e)}")
            pass

    converter.init_converter()
    log.debug(F"Original Action space size:{converter.n}")
    converter.filter_action(_filter_action)
    log.debug(F"Filtered Action space size:{converter.n}")
    if save_path is not None:
        log.debug(F"Saving filtered actions on {save_path}")
        my_path = os.path.join(save_path, AGENT_NAME)
        if not os.path.exists(my_path):
            os.makedirs(my_path)
        converter.save(my_path, "filtered_actions")
    return DiscreteActSpace(converter, action_list=converter.all_actions)


class CustomGymEnv(GymEnvWithHeuristics):
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, **kwargs):
        super().__init__(env_init, reward_cumul=reward_cumul, *args, **kwargs)
        self._safe_max_rho = safe_max_rho
        self.dn = self.init_env.action_space({})

    def heuristic_actions(self, g2op_obs, reward, done, info):
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


AGENT_NAME = "BasicExamplePPO"
SAVE_PATH = "../example_ppo"
config = {
    "core": {
        "logger": {
            "console": {
                "level": "INFO"
            },
            "file": {
                "level": "DEBUG",
                "filename": "execution_log.log",
                "mode": "w"
            }
        }
    },
    "common": {
        "name": AGENT_NAME,
        "save_path": SAVE_PATH,
        "env": {
            "simulator": grid2op,
            "env_class": grid2op.make,
            "env_kwargs": {
                "dataset": "l2rpn_wcci_2022",
                "difficulty": "competition",
                "reward_class":EpisodeDurationReward,
                "backend":LightSimBackend
            },
            "gymenv_class": CustomGymEnv
        },
        "action_space": {
            "class": create_action_space,
            "action_space_kwargs": {
                "save_path": SAVE_PATH,
                "load_path": SAVE_PATH
            }
        },
        "observation_space": {
            "class": BoxGymObsSpace,
            "observation_space_kwargs": {
                "attr_to_keep": ["day_of_week",
                                 "hour_of_day",
                                 "minute_of_hour",
                                 "prod_p",
                                 "prod_v",
                                 "load_p",
                                 "load_q",
                                 "actual_dispatch",
                                 "target_dispatch",
                                 "topo_vect",
                                 "time_before_cooldown_line",
                                 "time_before_cooldown_sub",
                                 "rho",
                                 "timestep_overflow",
                                 "line_status",
                                 "storage_power",
                                 "storage_charge"]
            }
        },
        "agent": {
            "maker": create_agent_sb3,
            "class": PPO,
            "net_kwargs":{
                "tensorboard_log": os.path.join(SAVE_PATH, "tensorboard_log"),
                "policy": MlpPolicy,
                "policy_kwargs":{
                    "net_arch":  [256,128, 64, 64, 32]
                }
            },
        },
        "training": False,
        "evaluation": False
    },
    "experiments": {
        "experiment_1": {
            "training": constants.TRAINING_DEFAULT,
            "training_kwargs": {
                "total_timesteps": 500000,
                "save_path": os.path.join(SAVE_PATH, AGENT_NAME),
                "callbacks": {
                    "save_progress": {
                        "class": CheckpointCallback,
                        "kwargs": {
                            "save_freq": 500000,
                            "save_path": SAVE_PATH,
                            "name_prefix": AGENT_NAME
                        }
                    }
                },
            },
            "evaluation": [
                {"evaluation": constants.EVALUATION_DEFAULT},
                {
                    "evaluation": constants.EVALUATION_GRID2OP,
                     "evaluation_kwargs": {
                         "logs_path": SAVE_PATH,
                         "nb_episode": 10,
                         "verbose":True,
                         "save_gif":True,
                     }
                 },
                {
                    "evaluation": constants.EVALUATION_L2RPN2022,
                    "evaluation_kwargs": {
                        "save_path": SAVE_PATH
                    }
                }
            ]
        }
    }
}


# Function to create the configuration for the main execution
# Its mandatory to have this function declared with this signature.
def get_config():
    return config


if __name__ == "__main__":
    main = AutoGrid.main(config, force_log="DEBUG")
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()
