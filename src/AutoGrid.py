import copy
import gc
import inspect
import json
import logging

from src.trainner.trainer import train_agent

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from src.evaluators.evaluator import evaluate_agent
from src.makers.maker import create_agent

import urllib3
from src.helpers import merge_dict
import logging.handlers
import os


class main(object):
    '''
    Main class, it receive the command line arguments and hold the main running loop.
    '''

    def __init__(self, input_config_json, force_log=False,
                 command_line_arguments=False):
        '''
        Initializer.

        Args:
            :input_config_file_name (string): File name of the configuration file to load
            :force_log (string): Force the log level
            :command_line_arguments (json): Extra arguments on the command line #CURRENTLY NOT IN USE
        '''
        self.default_params = {}
        if command_line_arguments != False:
            self.command_line_arguments = vars(command_line_arguments)
        else:
            self.command_line_arguments = {}
        self.force_debug_log = force_log
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.modules = {}
        self.config_json = input_config_json
        self.configure()

    def configure(self):
        '''
        Read the configuration file and prepare all the module instances for execution

        Calls:
            :func:`configure_logger`
        '''
        self.configure_logger()

    def configure_logger(self):
        '''
        Configure the loggin system.
        This can be configured using the "core" key in the configuration json

        >>> a simple example:
            "core": {
                "logger": {
                    "console": {
                        "level": "ERROR"
                    },
                    "file": {
                        "level": "DEBUG",
                        "filename": "execution_log.log",
                        "mode": "w"
                    }
                }
            },

            For more information on how to configure the logger please read the dedicated section :ref:`Logger configuration <logger_configuration_label>`
        '''
        log_format = self.config_json.get("core", {}).get("logger", {}).get("format", None)
        if log_format is None:
            log_format = {
                "fmt": "{asctime} | {levelname:7s} | {name:24s} | {lineno:<4n} | {message}",
                "style": "{",
                "datefmt": '%m-%d %H:%M'
            }
        logFormatter = logging.Formatter(**log_format)
        rootLogger = logging.getLogger()
        rootLogger.handlers = []

        log_config = self.config_json.get("core", {}).get("logger", {}).get("console", None)
        if log_config is None:
            log_config = {
                "level": logging.INFO,
            }
        if self.force_debug_log != False:
            log_config["level"] = self.force_debug_log
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(log_config.get("level"))
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

        log_file_config = self.config_json.get("core", {}).get("logger", {}).get("file", None)
        if log_file_config is None:
            log_file_config = {
                "level": logging.DEBUG,
                "filename": "execution_log.log",
                "maxBytes": 10485760,
                "backupCount": 10,
                "mode": "a"
            }
        if self.force_debug_log != False:
            log_file_config["level"] = self.force_debug_log
        log_path = os.path.dirname(log_file_config.get("filename"))
        if log_path:
            os.makedirs(log_path, exist_ok=True)

        fileHandler = logging.handlers.RotatingFileHandler(filename=log_file_config.get("filename"),
                                                           mode=log_file_config.get("mode", "a"),
                                                           maxBytes=log_file_config.get("maxBytes", 0),
                                                           backupCount=log_file_config.get("backupCount", 0),
                                                           encoding='utf-8')
        fileHandler.setLevel(log_file_config.get("level"))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        log.debug("Format logger configured [{}]".format(log_format))
        log.debug("Console logger configured [{}]".format(log_config))
        log.debug("File logger configured [{}]".format(log_file_config))

    def set_up_experiment_config(self, experiment_id):
        """
        experiment_config = merge_dict(self.config_json.get("common", {}),
                                       self.config_json.get("experiments").get(experiment_id),
                                       override=True, a_dict_name="common", b_dict_name=experiment_id)
"""
        experiment_config = merge_dict(self.config_json.get("experiments").get(experiment_id),
                                       self.config_json.get("common", {}),
                                       override=False, a_dict_name=experiment_id, b_dict_name="common")
        env_class = experiment_config.get("env").get("env_class")
        env_args = experiment_config.get("env").get("env_args",[])
        env_kwargs = experiment_config.get("env").get("env_kwargs",{})
        backend_class = experiment_config.get("env_backend_class",False)
        backend = env_kwargs.get("backend",False)
        if backend and (inspect.isclass(backend) or inspect.isfunction(backend)) :
            log.debug(F"Creating backend [{backend}]")
            env_kwargs["backend"] = backend()
        elif backend_class and (inspect.isclass(backend_class) or inspect.isfunction(backend_class)) :
            env_kwargs["backend"] = backend_class()

        #COPY the environment is not a good idea

        if inspect.isclass(env_class) or inspect.isfunction(env_class):
            log.debug(F'Creating a new environment using {env_class}({env_args, env_kwargs})')
            experiment_config["env"]["env"] = env_class(*env_args, **env_kwargs)
        elif self.config_json.get("experiments", {}).get(experiment_id, {}).get("env", {}).get("env", None) is None and \
                self.config_json.get("common",{}).get("env",{}).get("env",False) is not False :
            log.warning("THIS MODE OF HAVING THE ENVIROMENT WILL CRASH IF YOU HAVE MULTPLE EXPERIMENTS")
            log.warning("THIS MODE OF HAVING THE ENVIROMENT WILL CRASH IF YOU HAVE MULTPLE EXPERIMENTS")
            log.debug(F'Copy the environment from common config [{self.config_json.get("common").get("env").get("env")}] to experiment config')
            experiment_config["env"]["env"] = self.config_json.get("common").get("env").get("env").copy()

        experiment_config["name"] = experiment_config.get("name", experiment_id)

        return experiment_config

    def run(self):
        '''
        Main run loop.
        Execute all the configured Experiments in steps
        '''
        log.info("Starting execution.")
        for experiment_name in list(self.config_json.get("experiments")):
            log.info(F"Executing experiment [{experiment_name}]")
            experiment_config = self.set_up_experiment_config(experiment_name)
            experiment_config.get("env").get("env").reset()
            agent = create_agent(experiment_config)

            training = experiment_config.get("training", False)
            if training is not False:
                log.info(F"Training agent with [{training}]")
                agent = train_agent(agent, experiment_config)

            evaluation = experiment_config.get("evaluation", False)
            if evaluation is not False:
                log.info(F"Evaluating agent with [{evaluation}]")
                experiment_config.get("env").get("env").reset()
                evaluate_agent(agent, experiment_config)
                print("---------------------------------------------------------------------------------------------------------\n")
            experiment_config.get("env").get("env").close()

            del agent
            del experiment_config
            self.config_json.get("experiments").pop(experiment_name)
            gc.collect()

        log.info("Finished execution.")
