import sys
import os
from pathlib import Path

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
from src.helpers import create_experiment_gitignore


from src.makers.SB3 import create_agent_sb3
import citylearn
from citylearn.citylearn import CityLearnEnv

from src.envs.CityLearnEnv import EnvCityGym

from src import constants, AutoGrid
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy


class agent(PPO):
    def act(self, observation, reward, done):
        return self.predict(observation, deterministic=False)

#todo: integrate this function inside Autogrid, to be the same code regardles you are using citylearn or grid2op
def create_agentcreate_agent(experiment_config):
    env = EnvCityGym(experiment_config.get("env").get("env"))
    #env = experiment_config.get("env").get("env")
    return agent(policy=MlpPolicy,env= env, verbose=2, gamma=0.99,learning_rate=0.2,
                policy_kwargs={
                    "net_arch":  [256,128, 64, 64, 32]
                })

#TODO: THIS EXPERIMENT IS NOT JET WORKING
AGENT_NAME = "BasicExamplePPOcitylearn"
SAVE_PATH = "./example_ppo_citylearn"
config = {
    "experiments": {
        "experiment_1": {
            "name": AGENT_NAME,
            "save_path": SAVE_PATH,
            "env": {
                "simulator": citylearn,
                "env": CityLearnEnv(schema="citylearn_challenge_2020_climate_zone_1"),
                "gymenv_class":EnvCityGym,
            },
            "agent": {
                "maker": create_agentcreate_agent,
                "class":PPO,
            "net_kwargs":{
                "policy": MlpPolicy,
                "policy_kwargs":{
                    "net_arch":  [25, 25, 25, 25]
                }
            }
            },
            "training": constants.TRAINING_DEFAULT,
            "training_kwargs": {
                "total_timesteps": 100
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
