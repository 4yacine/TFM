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



from src.agents.Grid2OpSB3 import SB3AgentGrid2Op
import grid2op

from src import constants, AutoGrid
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from src.makers.SB3 import create_agent_sb3

AGENT_NAME = "BasicCustomExample"
SAVE_PATH = "./example_customagent"


class MySB3AgentClass(SB3AgentGrid2Op):
    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 nn_type,
                 nn_path=None,
                 nn_kwargs=None,
                 custom_load_dict=None,
                 gymenv=None,
                 iter_num=None,
                 ):
        super().__init__(g2op_action_space, gym_act_space, gym_obs_space,nn_type,
                         nn_path=nn_path,
                         nn_kwargs=nn_kwargs,
                         custom_load_dict=custom_load_dict,
                         gymenv=gymenv,
                         iter_num=iter_num
                         )
    def get_act(self, gym_obs, reward, done):
        """Retrieve the gym action from the gym observation and the reward.
        It only (for now) work for non recurrent policy.

        Parameters
        ----------
        gym_obs : gym observation
            The gym observation
        reward : ``float``
            the current reward
        done : ``bool``
            whether the episode is over or not.

        Returns
        -------
        gym action
            The gym action, that is processed in the :func:`GymAgent.act`
            to be used with grid2op
        """
        action, _ = self.nn_model.predict(gym_obs, deterministic=False)
        return action


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
                "agent":MySB3AgentClass,
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
