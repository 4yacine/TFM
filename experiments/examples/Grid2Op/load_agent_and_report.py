import sys
import os
from pathlib import Path


# This is incase you execute this example with a clone repository, so python can find AutoGrid
AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from lightsim2grid import LightSimBackend
import grid2op
from src.helpers import create_experiment_gitignore
from src import constants, AutoGrid
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from grid2op.gym_compat import BoxGymObsSpace, GymEnv

from complete_example import create_action_space
from src.makers.SB3 import create_agent_sb3

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
    "experiments": {
        "experiment_1": {
            "name": AGENT_NAME,
            "save_path": SAVE_PATH,
            "env": {
                "simulator": grid2op,
                "env": grid2op.make("l2rpn_wcci_2022", difficulty="competition",backend=LightSimBackend()),
                "gymenv_class": GymEnv
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
                    "attr_to_keep": ["day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v", "load_p",
                                     "load_q",
                                     "actual_dispatch", "target_dispatch", "topo_vect", "time_before_cooldown_line",
                                     "time_before_cooldown_sub", "rho", "timestep_overflow", "line_status",
                                     "storage_power", "storage_charge"]
                }
            },
            "agent": {
                "maker": create_agent_sb3,
                "class": PPO,
                "net_kwargs": {
                    "policy": MlpPolicy,
                },
                "load_path": os.path.join(SAVE_PATH,AGENT_NAME)  # HERE WE LOAD AN ALREADY TRAINED AGENT
            },
            "evaluation": [
                {
                    "evaluation": constants.EVALUATION_L2RPN2022,
                    "evaluation_kwargs":{
                        "save_path":"./example_ppo_l2rpn"
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
    main = AutoGrid.main(config)
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()
