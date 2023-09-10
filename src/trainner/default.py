import inspect
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def train_agent(agent,
            **trainner_kwargs):
    """
    Default training function that calls agent.train()
    :param agent: agent class to train
    :param trainner_kwargs: extra training arguments used by the agent
    :return: the trained agent
    """
    learn_signature = inspect.signature(agent.learn)
    for arg in list(trainner_kwargs.keys()):
        if arg not in learn_signature.parameters:
            trainner_kwargs.pop("save_path")
            log.warning(F"Removing '{arg}' from '{agent.learn}' function, since its not a valid parameter")
    agent.learn(**trainner_kwargs)
    return agent
