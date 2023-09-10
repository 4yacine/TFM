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



import grid2op

from lightsim2grid import LightSimBackend
from src import constants, AutoGrid
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from src.makers.SB3 import create_agent_sb3

AGENT_NAME = "BasicExamplePPO"
SAVE_PATH = "../example_ppo"
config = {
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
                "class":DiscreteActSpace,
            },
            "observation_space": {
                "class": BoxGymObsSpace,
            },
            "agent": {
                "maker": create_agent_sb3,
                "class": PPO,
                "net_kwargs":{
                    "policy": MlpPolicy,
                    "policy_kwargs":{
                        "net_arch": [25, 25, 25, 25]
                    }
                }
            },
            "training": constants.TRAINING_DEFAULT,
            "training_kwargs": {
                "total_timesteps": 100,
                "save_path": os.path.join(SAVE_PATH, AGENT_NAME)
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
