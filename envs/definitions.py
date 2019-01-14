from typing import List
from typing import Tuple


# Type aliases
Instruction                = List[int]
NaturalLanguageInstruction = Tuple[str, Instruction]


class InstructionEnvironmentDefinition:
    """
    The environment must have following functions:
        - reset
        - step
        - seed
        - goal_status
    The environment must have following parameters in constructor:
        - instruction
    """
    def __init__(self, env_constructor, **kwargs):
        self._env_constructor = env_constructor
        self._kwargs = kwargs
        self._name = self._get_name()

    def _get_name(self):
        return self._env_constructor(**self._kwargs).name()

    def build_env(self, instruction: Instruction):
        if not instruction:
            raise Exception("Instruction must be provided.")

        self._kwargs["instruction"] = instruction

        return self._env_constructor(**self._kwargs)

    def name(self):
        return self._name