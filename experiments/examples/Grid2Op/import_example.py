import sys
import os
from pathlib import Path
# This is incase you execute this example with a clone repository, so python can find AutoGrid
AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

from src.helpers import create_experiment_gitignore

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from src import constants, AutoGrid
AGENT_NAME = "BasicExamplePPO"
SAVE_PATH = "../example_ppo"

# Function to create the configuration for the main execution
# Its mandatory to have this function declared with this signature.
def get_config():
    from experiments.examples.Grid2Op.complete_example import config

    #WE override the experiments to change them
    config["experiments"] = {
        "experiment_1": {
            "evaluation": [
                {"evaluation": constants.EVALUATION_DEFAULT},
                {
                    "evaluation": constants.EVALUATION_GRID2OP,
                    "evaluation_kwargs": {
                        "logs_path": SAVE_PATH,
                        "nb_episode": 10,
                        "verbose": True,
                        "save_gif": True,
                    }
                }
            ]
        }
    }

    #We tell the agent to lad from path
    config["common"]["agent"]["load_path"]=os.path.join(SAVE_PATH,AGENT_NAME)
    return config


if __name__ == "__main__":
    main = AutoGrid.main(get_config(), force_log="DEBUG")
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()
