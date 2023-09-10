import logging

import gym

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import inspect
import os



def create_agent_sb3(experiment_config):

    # Get basic variables
    load_path = experiment_config.get("agent").get("load_path", None)
    name = experiment_config.get("name", "DAFAULT_NAME")
    env = experiment_config.get("env").get("env")
    Grid2OpAgentClass = experiment_config.get("agent").get("agent", False)
    if Grid2OpAgentClass is False:
        from src.agents.Grid2OpSB3 import SB3AgentGrid2Op
        Grid2OpAgentClass = SB3AgentGrid2Op

    log.debug(F"Environment [{env}]")
    _ = env.reset()
    if isinstance(env,gym.Env):
        env_gym = env
    else:
        # define the gym environment from the grid2op env
        gymenv_kwargs = experiment_config.get("env").get("gymenv_kwargs", {})

        gymenv_class = experiment_config.get("env").get("gymenv_class",None)
        if gymenv_class is None:
            log.error(F"gymenv_class config cant be None.")
        if inspect.isclass(gymenv_class) or inspect.isfunction(gymenv_class):
            env_gym = gymenv_class(env, **gymenv_kwargs)
        else:
            env_gym = gymenv_class
    log.debug(F"Gym Environment [{env_gym}]")

    if experiment_config.get("observation_space",False) is not False:
        # CREATE GYM OBSERVATION SPACE
        if env_gym.observation_space:
            env_gym.observation_space.close()
        obs_space_kwargs = experiment_config.get("observation_space").get("observation_space_kwargs", {})
        obs_space_class = experiment_config.get("observation_space").get("class")

        if inspect.isclass(obs_space_class) or inspect.isfunction(obs_space_class):
            log.debug(F"Creating new gym observation space using {obs_space_class} with arguments {obs_space_kwargs}")
            env_gym.observation_space = obs_space_class(
                env.observation_space,
                **obs_space_kwargs)
        else:
            log.debug("Using already existing gym observation space")
            env_gym.observation_space = obs_space_class
        log.debug(F"Gym observation_space [{env_gym.observation_space}]")

    if experiment_config.get("action_space",False) is not False:
        # CREATE GYM ACTION SPACE
        if env_gym.action_space:
            env_gym.action_space.close()
        action_space_kwargs = experiment_config.get("action_space").get("action_space_kwargs", {})
        action_space_class = experiment_config.get("action_space").get("class")

        if inspect.isclass(action_space_class) or inspect.isfunction(action_space_class):
            log.debug(F"Creating new gym action space using {action_space_class}  with arguments {action_space_kwargs}")
            env_gym.action_space = action_space_class(
                env.action_space,
                **action_space_kwargs)
        else:
            log.debug("Using already existing gym action space")
            env_gym.action_space = action_space_class
        log.debug(F"Gym action_space [{env_gym.action_space}]")

    model_class = experiment_config.get("agent").get("class")
    kwargs = experiment_config.get("agent").get("net_kwargs", {})
    logs_dir = experiment_config.get("agent").get("net_kwargs", {}).get("tensorboard_log", None)
    # define the policy
    if load_path is None:
        if logs_dir is not None:
            if not os.path.exists(logs_dir):
                os.mkdir(logs_dir)

        nn_kwargs = {
            "env": env_gym,
            "verbose": True,  # TODO add as parameter
            **kwargs
        }
        log.debug(F"Creating agent [{model_class}] with arguments {kwargs}")
        #TODO: Standarize agent class arguments
        agent = Grid2OpAgentClass(env.action_space,
                         env_gym.action_space,
                         env_gym.observation_space,
                         gymenv=env_gym,
                         nn_type=model_class,
                         nn_kwargs=nn_kwargs,
                         )
    else:
        agent = Grid2OpAgentClass(env.action_space,
                         env_gym.action_space,
                         env_gym.observation_space,
                         nn_type=model_class,
                         gymenv=env_gym,
                         nn_path=load_path
                         #iter_num="500000"
                         #TODO: Add option for Agent kwargs like iter_num
                         )
        log.info(F"Loaded agent from [{load_path}] ")

    #env_gym.close() #SINCE THE GYM_ENV IS USEED IN CASE OF HEURISTIC HACTONS THE ENV CANT BE CLOSED
    #TODO - THE ENV MUST BE CLOSE SOMEWHERE ELSE AT THE END OF THE EXPERIMENT
    return agent
