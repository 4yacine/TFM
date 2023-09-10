import sys
import os
from pathlib import Path

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from experiments.L2RPN import common

from src.helpers import create_experiment_gitignore
from src import constants, AutoGrid
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy as dqnMlpPolicy

SAVE_PATH = "./agent"
AGENT_NAME = "L2RPN_DQN"
# Function to create the configuration for the main execution
# Its mandatory to have this function declared with this signature.
def get_config():
    config = common.get_config()
    config["common"]["training"]=False

    config["common"]["name"] = AGENT_NAME
    config["common"]["training_kwargs"]["callbacks"]["save_progress"]["kwargs"]["name_prefix"] = AGENT_NAME
    config["common"]["agent"]["class"] = DQN
    config["common"]["agent"]["net_kwargs"]["policy"] = dqnMlpPolicy
    config["common"]["action_space"]["action_space_kwargs"] = {
            "save_path": os.path.join(SAVE_PATH,AGENT_NAME),
            "load_path": os.path.join(SAVE_PATH,AGENT_NAME),
    }

    for file in os.listdir(SAVE_PATH):
        if file.startswith(AGENT_NAME) and file.endswith(".zip") and os.path.isfile(os.path.join(SAVE_PATH,file)):
            file_name = os.path.splitext(file)[0]
            config["experiments"][file_name]={
                "name":file_name,
                "agent":{
                    "load_path":os.path.join(SAVE_PATH,file_name),
                },
                "evaluation_kwargs": {
                    "save_path": os.path.join(SAVE_PATH, file_name),
                }
            }

    return config


if __name__ == "__main__":
    main = AutoGrid.main(get_config())
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()
