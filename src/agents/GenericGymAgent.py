
import copy
import inspect
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class GenericGymAgent(object):

    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 nn_type,
                 nn_path=None,
                 nn_kwargs=None,
                 custom_load_dict=None,
                 gymenv=None,
                 eval_env = None,
                 _check_both_set=True,
                 _check_none_set=True):
        self._nn_type = nn_type
        self.gymenv = gymenv
        self.eval_env = eval_env

        if _check_none_set and (nn_path is None and nn_kwargs is None):
            raise RuntimeError("Impossible to build a GymAgent without providing at "
                               "least one of `nn_path` (to load the agent from disk) "
                               "or `nn_kwargs` (to create the underlying agent).")
        if _check_both_set and (nn_path is not None and nn_kwargs is not None):
            raise RuntimeError("Impossible to build a GymAgent by providing both "
                               "`nn_path` (*ie* you want load the agent from disk) "
                               "and `nn_kwargs` (*ie* you want to create the underlying agent from these "
                               "parameters).")
        if nn_path is not None:
            self._nn_path = nn_path
        else:
            self._nn_path = None

        if nn_kwargs is not None:
            self._nn_kwargs = copy.deepcopy(nn_kwargs)
        else:
            self._nn_kwargs = None

        self.nn_model = None
        if nn_path is not None:
            self.load()
        else:
            self.build()

    def get_act(self, gym_obs, reward, done):
        """
        retrieve the action from the NN model
        """
        action, _ = self.nn_model.predict(gym_obs, deterministic=False)
        return action

    def load(self):
        """
        Load the NN model.

        In the case of a PPO agent, this is equivalent to perform the:

        .. code-block:: python

            PPO.load(nn_path)
        """
        custom_objects = {"action_space": self._gym_act_space,
                          "observation_space": self._gym_obs_space}
        for key, val in self.custom_load_dict.items():
            custom_objects[key] = val
        path_load = self._nn_path
        if self._iter_num is not None:
            path_load = path_load + f"_{self._iter_num}_steps"
        log.debug(F"loading agent from [{path_load}]")
        self.nn_model = self._nn_type.load(path_load,
                                           custom_objects=custom_objects,
                                           env=self.gymenv)

    def build(self):
        """Create the underlying NN model from scratch.

        In the case of a PPO agent, this is equivalent to perform the:

        .. code-block:: python

            PPO(**nn_kwargs)
        """
        self.nn_model = self._nn_type(**self._nn_kwargs)



    def act(self, observation, reward, done):
        gym_act = self.get_act(observation, reward, done)
        return gym_act


    def learn(self,
          total_timesteps=1,
          save_path=None,
          callbacks={},
          **learn_kwargs):

        list_of_callbacks=[]
        for callback_name,callback_data in callbacks.items():
            callback_class=callback_data.get("class")
            if inspect.isclass(callback_class) or inspect.isfunction(callback_class):
                kwargs = callback_data.get("kwargs",{})
                callback = callback_class(**kwargs)
            else:
                callback= callback_class
            log.info(F"Configured callback [{callback_name}] using class [{callback_class}]")
            list_of_callbacks.append(callback)
        if learn_kwargs is None:
            learn_kwargs={}
        # train it
        self.nn_model.learn(total_timesteps=total_timesteps,
                             callback=list_of_callbacks,
                             eval_env=self.eval_env,
                            **learn_kwargs
                             )

        # save it
        if save_path is not None:
            self.nn_model.save(save_path)