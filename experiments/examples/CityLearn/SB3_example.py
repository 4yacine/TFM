import sys
import os
from pathlib import Path

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from src.helpers import create_experiment_gitignore

from src import constants, AutoGrid
from stable_baselines3.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

def create_agent(experiment_config):
    env = experiment_config.get("env").get("env")
    return SAC('MlpPolicy', env)

AGENT_NAME = "BasicExampleSB3PPOcitylearn"
SAVE_PATH = "./example_SBr_PPO_citylearn"
dataset_name = 'citylearn_challenge_2022_phase_1'

config = {
    "experiments": {
        "experiment_1": {
            "name": AGENT_NAME,
            "save_path": SAVE_PATH,
            "env": {
                "env": StableBaselines3Wrapper(NormalizedObservationWrapper(CityLearnEnv(dataset_name, central_agent=True)))
            },
            "agent": {
                "maker": create_agent,
            },
            "training": constants.TRAINING_DEFAULT,
            "training_kwargs": {
                "total_timesteps": 30000
            },
            "evaluation":  constants.EVALUATION_CITYLEARN_ENV
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
