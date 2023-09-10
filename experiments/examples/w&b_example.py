import sys
import os
from pathlib import Path

from grid2op.Converter import IdToAct
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from src.envs.gymenv_heuristics import GymEnvWithHeuristics

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

from src.helpers import create_experiment_gitignore

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


import grid2op
import numpy as np
from src import constants, AutoGrid
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv
from src.makers.SB3 import create_agent_sb3
from stable_baselines3.common.monitor import Monitor

import wandb


AGENT_NAME = "BasicExamplePPO"
SAVE_PATH = "./example_ppo"

from wandb.integration.sb3 import WandbCallback


def make_env(grid2openv):
    env = CustomGymEnv(grid2openv)
    #env = Monitor(env)  # record stats such as returns
    return env


def _filter_action(action):
    MAX_ELEM = 2
    act_dict = action.impact_on_objects()
    # Remove Curtailment
    if act_dict["curtailment"]["changed"] or action._modif_curtailment or len(act_dict["curtailment"]["limit"]) > 0:
        return False
    elem = 0
    elem += act_dict["force_line"]["reconnections"]["count"]
    elem += act_dict["force_line"]["disconnections"]["count"]
    elem += act_dict["switch_line"]["count"]
    elem += len(act_dict["topology"]["bus_switch"])
    elem += len(act_dict["topology"]["assigned_bus"])
    elem += len(act_dict["topology"]["disconnect_bus"])
    elem += len(act_dict["redispatch"]["generators"])

    if elem <= MAX_ELEM:
        return True
    return False


def create_action_space(env_action_space, load_path=None, save_path=None):
    converter = IdToAct(env_action_space)
    if load_path is not None:
        try:
            my_path = os.path.join(load_path, AGENT_NAME)
            converter.init_converter(all_actions=os.path.join(my_path, "filtered_actions.npy"))
            log.debug(F"Loaded Action space size:{converter.n} from folder [{my_path}]")
            return DiscreteActSpace(converter, action_list=converter.all_actions)
        except FileNotFoundError as e:
            log.warning(F"Could not load filtered action space, one will be created now. Error: {str(e)}")
            pass

    converter.init_converter()
    log.debug(F"Original Action space size:{converter.n}")
    converter.filter_action(_filter_action)
    log.debug(F"Filtered Action space size:{converter.n}")
    if save_path is not None:
        log.debug(F"Saving filtered actions on {save_path}")
        my_path = os.path.join(save_path, AGENT_NAME)
        if not os.path.exists(my_path):
            os.makedirs(my_path)
        converter.save(my_path, "filtered_actions")
    return DiscreteActSpace(converter, action_list=converter.all_actions)

class CustomGymEnv(GymEnvWithHeuristics):
    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, **kwargs):
        super().__init__(env_init, reward_cumul=reward_cumul, *args, **kwargs)
        self._safe_max_rho = safe_max_rho
        self.dn = self.init_env.action_space({})

    def heuristic_actions(self, g2op_obs, reward, done, info):
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        res = []
        if np.any(to_reco):
            # reconnect something if it can be
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
                res.append(g2op_act)
        elif g2op_obs.rho.max() <= self._safe_max_rho:
            # play do nothing if there is "no problem" according to the "rule of thumb"
            res = [self.init_env.action_space()]
        return res

    def step(self, gym_action):

        g2op_act = self.action_space.from_gym(gym_action)

        _, sim_reward_act, _, _ = self.init_env.simulate(g2op_act)
        _, sim_reward_dn, _, _ = self.init_env.simulate(self.dn)
        if sim_reward_dn > sim_reward_act:
            g2op_act = self.dn

        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        if not done:
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, float(reward), done, info



config = {
    "experiments": {
        "experiment_1": {
            "name": AGENT_NAME,
            "save_path": SAVE_PATH,
            "env": {
                "simulator": grid2op,
                "env": grid2op.make("l2rpn_wcci_2022", difficulty="competition"),
                "gymenv_class":make_env,
            },
            "action_space": {
                "class":create_action_space,
            },
            "observation_space": {
                "class": BoxGymObsSpace,
            },
            "agent": {
                "maker": create_agent_sb3,
                "class": PPO,
                "net_kwargs":{
                    "tensorboard_log": os.path.join(SAVE_PATH, "tensorboard_log"),
                    "policy": MlpPolicy,
                    "policy_kwargs":{
                        "net_arch":  [256,128, 64, 64, 32]
                    }
                }
            },
            "training": constants.TRAINING_DEFAULT,
            "training_kwargs": {
                "total_timesteps": 25000,
                "save_path": os.path.join(SAVE_PATH, AGENT_NAME),
                "callbacks": {
                    "wnb_progress": {
                        "class": WandbCallback,
                        "kwargs": {
                            "gradient_save_freq": 100,
                            "model_save_path": os.path.join(SAVE_PATH, AGENT_NAME),
                            "verbose": 2
                        }
                    }
                },
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
    #wandb.tensorboard.patch(root_logdir=os.path.join(SAVE_PATH, "tensorboard_log"))
    run = wandb.init(
        project="sb3",
        config={
            "policy_type": "MlpPolicy",
            "total_timesteps": 25000,
            "env_name": "l2rpn_wcci_2022",
        },
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    main = AutoGrid.main(config, force_log="DEBUG")
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()

    run.finish()
