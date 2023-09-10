import sys
import os
from pathlib import Path

from grid2op.gym_compat import MultiDiscreteActSpace

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

from src.helpers import create_experiment_gitignore

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from experiments.L2RPN import common

from src import AutoGrid
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

AGENT_NAME = "L2RPN_PPO"
SAVE_PATH = "./agent"
# Function to create the configuration for the main execution
# Its mandatory to have this function declared with this signature.
def get_config():
    config = common.get_config()
    config["common"]["name"] = AGENT_NAME
    config["common"]["training_kwargs"]["callbacks"]["save_progress"]["kwargs"]["name_prefix"] = AGENT_NAME
    config["common"]["agent"]["class"] = PPO
    config["common"]["agent"]["net_kwargs"]["policy"] = MlpPolicy
    config["common"]["action_space"]= {
                "class" : MultiDiscreteActSpace
        }
    config["experiments"]["experiment_1"]={}
    return config


if __name__ == "__main__":
    main = AutoGrid.main(get_config(), force_log="DEBUG")
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()
