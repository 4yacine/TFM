import sys
import os
from pathlib import Path

# This is incase you execute this example with a cloned repository, so python can find AutoGrid
from agents.GenericGymAgent import GenericGymAgent
from makers.SB3 import create_agent_sb3

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
from src.helpers import create_experiment_gitignore

import gym
from src import constants, AutoGrid
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy


AGENT_NAME = "BasicExamplePPOANM"
SAVE_PATH = "./example_ppo_anm"
config = {
    "experiments": {
        "experiment_1": {
            "name": AGENT_NAME,
            "save_path": SAVE_PATH,
            "env": {
                "env": gym.make('gym_anm:ANM6Easy-v0')
            },
            "agent": {
                "maker": create_agent_sb3,
                "class": PPO,
                "agent":GenericGymAgent,
                "net_kwargs":{
                    "policy": MlpPolicy,
                    "policy_kwargs":{
                        #"net_arch": [25, 25, 25, 25]
                    }
                }
            },
            "training": constants.TRAINING_DEFAULT,
            "training_kwargs": {
                "total_timesteps": 50000000,
                "save_path": os.path.join(SAVE_PATH, AGENT_NAME),
            },
            "evaluation":  constants.EVALUATION_DEFAULT,
            "evaluation_kwargs":{
                "verbose":2,
                "max_steps":1000
            }
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



"""
"total_timesteps": 500.000,
Mean score over [100 episodes] : 
score     -79.809524
steps    1000.000000
"""
"""
"total_timesteps": 5.000.000,
Mean score over [100 episodes] : 
score     -37.915391
steps    1000.000000
"""