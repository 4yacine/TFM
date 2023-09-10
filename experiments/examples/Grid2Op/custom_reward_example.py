import sys
import os
from pathlib import Path

# This is incase you execute this example with a clone repository, so python can find AutoGrid
AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


from grid2op.Reward import EpisodeDurationReward
from lightsim2grid import LightSimBackend
import grid2op
from src.helpers import create_experiment_gitignore
from src import constants, AutoGrid
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from src.makers.SB3 import create_agent_sb3

AGENT_NAME = "BasicExamplePPO"
SAVE_PATH = "../example_ppo"


class RewardClass(grid2op.Reward.BaseReward):
    """
    Class to control the reward used by the environment when your agent is being assessed.
        # You can look at the grid2op documentation to have example on definition of rewards
        # https://grid2op.readthedocs.io/en/latest/reward.html
    """

    def __init__(self):
        self.n_steps = 0
        grid2op.Reward.BaseReward.__init__(self)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            self.n_steps = 0
            return -10
        else:
            self.n_steps+=1
            return self.n_steps


config = {
    "experiments": {
        "experiment_1": {
            "name": AGENT_NAME,
            "save_path": SAVE_PATH,
            "env": {
                "simulator": grid2op,
                "env_class":grid2op.make,
                "env_kwargs":{
                    "dataset":"l2rpn_wcci_2022",
                    "difficulty":"competition",
                    "reward_class":EpisodeDurationReward,
                    "backend":LightSimBackend()
                },
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
