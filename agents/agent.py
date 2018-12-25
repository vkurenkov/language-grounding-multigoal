import gym

from tensorboardX     import SummaryWriter
from typing           import Optional, Dict, List
from envs.definitions import Instruction
from envs.definitions import InstructionEnvironmentDefinition


class Agent:
    """
    - Discrete actions
    - Stateful (may maintain state over time)
    - Episodic
    """

    def log_init(self, summary_writer: SummaryWriter) -> None:
        self._log_writer = summary_writer
        

    def train_init(self, 
            env_definition: InstructionEnvironmentDefinition, 
            training_instructions: List[Instruction]) -> None:
        raise NotImplementedError()

    def train_step(self) -> None:
        raise NotImplementedError()

    def train_num_steps(self) -> int:
        raise NotImplementedError()

    def train_is_done(self) -> bool:
        raise NotImplementedError()

    
    def reset(self) -> None:
        """
        Should reset to the initial state for an episode.
        """
        raise NotImplementedError()

    def act(self, observation, instruction: Instruction, env: Optional[gym.Env] = None) -> Optional[int]:
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