import logging

from grid2op.Agent import BaseAgent

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class Do_nothing_agent(BaseAgent):
    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 gymenv=None,
                 *args,
                 **kwargs
                 ):
        super().__init__(g2op_action_space)
        self.do_nothing_action = g2op_action_space({})

    def act(self, observation, reward, done):
        return self.do_nothing_action

    def build(self):
        pass

    def learn(self,
              total_timesteps=1,
              save_path=None,
              callbacks={},
              **learn_kwargs):
        pass
