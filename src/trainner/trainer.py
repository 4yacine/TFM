import inspect
import logging
import os

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from src import constants
from src.trainner import default


def train_agent(agent,experiment_config):
    trainner = experiment_config.get("training")
    trainner_kwargs = experiment_config.get("training_kwargs",{})
    log.debug(F"Training agent [{agent}] with trainner: {trainner}({trainner_kwargs})")

    save_path = trainner_kwargs.get("save_path", None)
    global_save_path = experiment_config.get("save_path", None)
    experiment_name = experiment_config.get('name')
    if save_path is None and global_save_path:
        trainner_kwargs["save_path"] = os.path.join(global_save_path, experiment_name)

    if trainner in [constants.TRAINING_DEFAULT]:
        return default.train_agent(
            agent,
            **trainner_kwargs
        )
    elif inspect.isclass(trainner):
        log.warning(F"Class-base trainer are not fully designed")
        return trainner(agent, **trainner_kwargs).train()
    elif inspect.isfunction(trainner):
        return trainner(agent, **trainner_kwargs)
    else:
        log.warning(F"[{trainner}] is not a valid training method or class.")
