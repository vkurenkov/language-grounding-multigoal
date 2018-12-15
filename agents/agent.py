import gym
from typing import Optional, Dict


class Agent:
    def log_init(self, summary_writer) -> None:
        self._log_writer = summary_writer
        
    def train_init(self, env_definition) -> None:
        raise NotImplementedError()

    def train_step(self) -> None:
        raise NotImplementedError()

    def train_num_steps(self) -> int:
        raise NotImplementedError()

    def train_is_done(self) -> bool:
        raise NotImplementedError()

    def act(self, observation, env: Optional[gym.Env] = None) -> Optional[int]:
        raise NotImplementedError()

    def parameters(self) -> Dict:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()

    def __getstate__(self):
        '''
        Remove log writer from pickling.
        '''
        state = self.__dict__.copy()
        del state['_log_writer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)