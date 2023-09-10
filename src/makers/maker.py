import inspect
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def create_agent(experiment_config):
    agent_maker=experiment_config.get("agent").get("maker")
    if inspect.isclass(agent_maker) or inspect.isfunction(agent_maker):
        return agent_maker(experiment_config)
    else:
        log.warning(F"[{agent_maker}] is not a valid agent maker method or class.")
