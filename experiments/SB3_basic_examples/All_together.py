import sys
import os
from pathlib import Path
AutoGridPath = os.path.dirname(Path(__file__).parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

from src.helpers import create_experiment_gitignore

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


import grid2op

from src import constants, AutoGrid
from stable_baselines3 import A2C, DDPG, SAC, TD3, DQN, PPO
from stable_baselines3.a2c import MlpPolicy as a2cMlpPolicy
from stable_baselines3.ddpg import MlpPolicy as ddpgMlpPolicy
from stable_baselines3.sac import MlpPolicy as sacMlpPolicy
from stable_baselines3.td3 import MlpPolicy as td3MlpPolicy
from stable_baselines3.dqn import MlpPolicy as dqnMlpPolicy
from stable_baselines3.ppo import MlpPolicy as ppoMlpPolicy
from grid2op.gym_compat import BoxGymObsSpace, GymEnv, BoxGymActSpace, DiscreteActSpace
from src.makers.SB3 import create_agent_sb3

SAVE_PATH = "./agents"
config = {
    "common": {
        "save_path": SAVE_PATH,
        "env": {
            "simulator": grid2op,
            "env": grid2op.make("l2rpn_wcci_2022", difficulty="competition"),
            "gymenv_class": GymEnv
        },
        "action_space": {
            "class":BoxGymActSpace,
        },
        "observation_space": {
            "class": BoxGymObsSpace,
            "observation_space_kwargs":{
                "attr_to_keep":[ "gen_p",
                                "gen_q",
                                "gen_v",
                                 "load_p",
                                "load_q",
                                "load_v",
                                "curtailment"]
                }
        },
        "agent": {
            "maker": create_agent_sb3,
            "net_kwargs":{
                "policy_kwargs": {
                    "net_arch": [64, 64, 64, 64],
                }
            }
        },
        "training": constants.TRAINING_DEFAULT,
        "training_kwargs": {
            "total_timesteps": 1000000,
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
            "name": "L2RPN",
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
                "save_path":os.path.join(SAVE_PATH,"L2RPN")
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
    main = AutoGrid.main(config)
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()
