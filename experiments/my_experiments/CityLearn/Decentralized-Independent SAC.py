import sys
import os
from pathlib import Path

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from src.helpers import create_experiment_gitignore
import citylearn
from citylearn.citylearn import CityLearnEnv


from src import constants, AutoGrid
from citylearn.agents.sac import SAC
from citylearn.agents.marlisa import MARLISA
from citylearn.agents.rbc import RBC

from typing import List
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction

def create_agent_SAC(experiment_config):
    env = experiment_config.get("env").get("env")
    return SAC(env)

def create_agent_MARLISA(experiment_config):
    env = experiment_config.get("env").get("env")
    return MARLISA(env)

def create_agent_RBC(experiment_config):
    env = experiment_config.get("env").get("env")
    return RBC(env)

class CustomReward(RewardFunction):
    def __init__(self, env: CityLearnEnv):
        super().__init__(env)

    def calculate(self) -> List[float]:
        #reward = [b.net_electricity_consumption[-1] for b in self.env.buildings]
        #reward = [(b.net_electricity_consumption[-1]/b.net_electricity_consumption_without_storage[-1]).sum() for b in self.env.buildings]
        
        #reward = [b.net_electricity_consumption_emission[-1] for b in self.env.buildings]
        
        #reward = [0,0,0,0,0] aqui se ve que los cmabios funcionan bien Yacine
        
        reward = [min((b.net_electricity_consumption_emission[b.time_step] + 
                      b.net_electricity_consumption_cost[b.time_step])*(-1), 0) for b in self.env.buildings]

        return reward

def create_env():
    env = CityLearnEnv(schema="citylearn_challenge_2022_phase_3")
    env.reward_function = CustomReward(env=env)
    return env

config = {
    "common":{
        "env": {
            "env_class": create_env,
        },
        "training": constants.TRAINING_DEFAULT,
        "training_kwargs": {
            "episodes": 25
        },
        "evaluation":  constants.EVALUATION_CITYLEARN_ENV
    },
    "experiments": {
        "experiment_SAC": {
            "name": "Agente_SAC",
            "agent": {
                "maker": create_agent_SAC,
            },
        },
        "experiment_MARLISA": {
            "name": "Agente_MARLISA",
            "agent": {
                "maker": create_agent_MARLISA,
            },
        },
        "experiment_RBC": {
            "name": "Agente_RBC",
            "agent": {
                "maker": create_agent_RBC,
            },
        }
    }
}


# Function to create the configuration for the main execution
# Its mandatory to have this function declared with this signature.
def get_config():
    return config


if __name__ == "__main__":
    main = AutoGrid.main(config, force_log="DEBUG")
    log.info(F"Executing experiment file {__file__}")
    main.run()
