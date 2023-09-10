import os
import sys
from pathlib import Path

from lightsim2grid import LightSimBackend

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import os

import grid2op
from grid2op.Converter import IdToAct

from src import constants
from stable_baselines3.common.callbacks import CheckpointCallback
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, GymEnv, MultiDiscreteActSpace
from src.makers.SB3 import create_agent_sb3


def _filter_action(action):
    MAX_ELEM = 3
    act_dict = action.impact_on_objects()
    # Remove Curtailment
    # if act_dict["curtailment"]["changed"] or action._modif_curtailment or len(act_dict["curtailment"]["limit"]) > 0:
    #    return False
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


def create_action_space(env_action_space, load_path=None, save_path=None, file_name="filtered_actions"):
    converter = IdToAct(env_action_space)
    if load_path is not None:
        try:
            converter.init_converter(all_actions=os.path.join(load_path, F"{file_name}.npy"))
            log.debug(F"Loaded Action space size:{converter.n} from folder [{load_path}]")
            return DiscreteActSpace(converter, action_list=converter.all_actions)
        except FileNotFoundError as e:
            log.warning(F"Could not load filtered action space, one will be created now. Error: {str(e)}")
            pass

    converter.init_converter()
    log.debug(F"Original Action space size:{converter.n}")
    #converter.filter_action(_filter_action)
    log.debug(F"Filtered Action space size:{converter.n}")
    if save_path is not None:
        log.debug(F"Saving filtered actions on {save_path}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        converter.save(save_path, file_name)
    return DiscreteActSpace(converter, action_list=converter.all_actions)


SAVE_PATH = "./agent"
config = {
    "core": {
        "logger": {
            "console": {
                "level": "INFO"
            },
            "file": {
                "level": "DEBUG",
                "filename": "execution_log.log",
                "mode": "w"
            }
        }
    },
    "common": {
        "name": None,
        "save_path": SAVE_PATH,
        "env": {
            "simulator": grid2op,
            "env": grid2op.make("l2rpn_wcci_2022", difficulty="competition",backend=LightSimBackend()),
            "gymenv_class": GymEnv
        },
        "action_space": {
            "class": MultiDiscreteActSpace
        },
        "observation_space": {
            "class": BoxGymObsSpace,
            "observation_space_kwargs": {
                "attr_to_keep": [
                    "hour_of_day",
                    "day_of_week",
                    "gen_p",
                    "gen_p_before_curtail",
                    "gen_q",
                    "gen_v",
                    "gen_margin_up",
                    "gen_margin_down",
                    "load_p",
                    "load_q",
                    "load_v",
                    "p_or",
                    "q_or",
                    "v_or",
                    "a_or",
                    "p_ex",
                    "q_ex",
                    "v_ex",
                    "a_ex",
                    "rho",
                    "line_status",
                    "timestep_overflow",
                    "topo_vect",
                    "time_before_cooldown_line",
                    "time_before_cooldown_sub",
                    "time_next_maintenance",
                    "duration_next_maintenance",
                    "target_dispatch",
                    "actual_dispatch",
                    "storage_charge",
                    "storage_power_target",
                    "storage_power",
                    "curtailment",
                    "curtailment_limit",
                    "curtailment_limit_effective",
                    "thermal_limit",
                    "theta_or",
                    "theta_ex",
                    "load_theta",
                    "gen_theta"]
            }
        },
        "agent": {
            "maker": create_agent_sb3,
            "net_kwargs": {
                "tensorboard_log": os.path.join(SAVE_PATH, "tensorboard_log"),
                "policy_kwargs": {
                    "net_arch": [256, 256, 256, 256]
                }
            },
        },
        "training": constants.TRAINING_DEFAULT,
        "training_kwargs": {
            "total_timesteps": 100000,
            "save_path": SAVE_PATH,
            "callbacks": {
                "save_progress": {
                    "class": CheckpointCallback,
                    "kwargs": {
                        "save_freq": 50000,
                        "save_path": SAVE_PATH,
                        "name_prefix": None
                    }
                }
            }
        },
        "evaluation": constants.EVALUATION_L2RPN2022,
        "evaluation_kwargs": {
            "save_path": SAVE_PATH
        }
    },
    "experiments": {
    }
}


# Function to create the configuration for the main execution
# Its mandatory to have this function declared with this signature.
def get_config():
    return config


if __name__ == "__main__":
    raise NotImplementedError("This file is a common definition and shall not be executed by itself")
