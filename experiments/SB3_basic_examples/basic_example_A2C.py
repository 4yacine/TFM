import sys
import os
from pathlib import Path

from lightsim2grid import LightSimBackend

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

from src.helpers import create_experiment_gitignore

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


import grid2op

from src import constants, AutoGrid
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy as a2cMlpPolicy
from grid2op.gym_compat import BoxGymObsSpace, GymEnv, BoxGymActSpace
from src.makers.SB3 import create_agent_sb3

AGENT_NAME = "BasicExampleA2C"
SAVE_PATH = "./agents"
config = {
    "common": {
        "name": AGENT_NAME,
        "save_path": SAVE_PATH,
        "env": {
            "simulator": grid2op,
            "env": grid2op.make("l2rpn_wcci_2022", difficulty="competition",backend=LightSimBackend()),
            "gymenv_class": GymEnv
        },
        "action_space": {
            "class":BoxGymActSpace,
            "action_space_kwargs":{
                "attr_to_keep":[
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
                    "gen_p_before_curtail",
                    "gen_q",
                    "gen_v",
                    "gen_margin_up",
                    "gen_margin_down",
                    "load_p",
                    "load_q",
                    "load_v",
                    "p_or",
                    "q_or",
                    "v_or",
                    "a_or",
                    "p_ex",
                    "q_ex",
                    "v_ex",
                    "a_ex",
                    "rho",
                    "curtailment"]
                }
        },
        "agent": {
            "maker": create_agent_sb3,
            "class": A2C,
            "net_kwargs":{
                "tensorboard_log": os.path.join(SAVE_PATH, "tensorboard_log"),
                "policy": a2cMlpPolicy
            }
        },
        "training": False,
        "evaluation": False
    },
    "experiments": {
        "experiment_1": {
            "training": constants.TRAINING_DEFAULT,
            "training_kwargs": {
                "total_timesteps": 100000,
                "save_path": os.path.join(SAVE_PATH, AGENT_NAME),
            },
            "evaluation":  constants.EVALUATION_DEFAULT,
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
