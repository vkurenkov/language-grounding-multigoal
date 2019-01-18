import warnings

from typing                    import Optional, Dict, List
from agents.agent              import Agent
from envs.gridworld_simple.env import FindItemsEnv
from envs.definitions          import NaturalLanguageInstruction
from envs.definitions          import InstructionEnvironmentDefinition


class PerfectAgent(Agent):
    def log_init(self, summary_writer) -> None:
        pass

    def train_init(self, 
            env_definition: InstructionEnvironmentDefinition, 
            training_instructions: List[NaturalLanguageInstruction]) -> None:
        pass

    def train_step(self) -> None:
        pass

    def train_num_steps(self) -> int:
        return 0

    def train_is_done(self) -> bool:
        return True

    def reset(self) -> None:
        """
        Nothing to reset here.
        """
        pass

    def act(self, observation, instruction: NaturalLanguageInstruction, env: Optional[FindItemsEnv]) -> Optional[int]:
        if not env:
            raise Exception("Cannot act without information about an environment.")

        agent_pos, *_ = observation
        path = env._shortest_paths.get_path(agent_pos, env._cur_target_item())

        if not path:
            warnings.warn("I could not find a path to the target item.")
            return None
        elif len(path) == 0:
            warnings.warn("I am already at the target item.")
            return None
        else:
            move_pos = path[0]
            move_dir = (move_pos[0] - agent_pos[0], move_pos[1] - agent_pos[1])
            if move_dir == (1, 0):
                return FindItemsEnv.ACTION_MOVE_RIGHT
            elif move_dir == (-1, 0):
                return FindItemsEnv.ACTION_MOVE_LEFT
            elif move_dir == (0, 1):
                return FindItemsEnv.ACTION_MOVE_UP
            elif move_dir == (0, -1):
                return FindItemsEnv.ACTION_MOVE_DOWN
            else:
                raise Exception("This should not happen.")

    def parameters(self) -> Dict:
        return {}

    def name(self) -> str:
        return "perfect"