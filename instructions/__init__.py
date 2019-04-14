import json
import os
import random
import numpy as np

from envs.definitions                              import NaturalLanguageInstruction
from envs.definitions                              import Instruction
from envs.gridworld_simple.instructions.tokenizer  import InstructionTokenizer
from typing                                        import List, Tuple
from utils.training                                import fix_random_seeds

def get_this_file_path() -> str:
    return os.path.dirname(__file__)

def read_instructions_from_disk(path: str) -> List[NaturalLanguageInstruction]:
    dataset_path = os.path.join(path)
    with open(dataset_path, mode="r") as f:
        dataset = json.load(f)

    return [(instruction["raw"], instruction["objects_real_order"]) for instruction in dataset["instructions"]]

def get_instructions(level: int, max_train_subgoals: int, unseen_proportion: float, seed: int):
    """
    Divides the specified level dataset into 3 parts: training instructions, unseen combinations, and higher number of subgoals.

    :param max_train_subgoals: Maximum number of available sub-goals during training. Others will go to the 3rd part. 
    :param unseen_proportion: Split between training instructions and unseen combinations.
    :returns: training instructions, unseen combinations, higher number of sub-goals
    """
    # We want our splits to be reproducible
    fix_random_seeds(seed)

    dataset_path = os.path.join(get_this_file_path(), "level{}.json".format(level))
    instructions = read_instructions_from_disk(dataset_path)

    print("Level{}: Total {} instructions.".format(level, len(instructions)))

    higher_num_subgoals = []
    training            = []
    for instruction in instructions:
        num_subgoals = len(instruction[1])
        if num_subgoals > max_train_subgoals:
            higher_num_subgoals.append(instruction)
        else:
            training.append(instruction)

    training = np.array(training, dtype=object)
    np.random.shuffle(training)

    num_unseen = int(len(training) * unseen_proportion)
    unseen     = training[:num_unseen]
    training   = training[num_unseen:]

    print("Level{}: Training {} instructions.".format(level, len(training)))
    print("Level{}: Unseen {} instructions.".format(level, len(unseen)))
    print("Level{}: Higher subgoals {} instructions.".format(level, len(higher_num_subgoals)))

    return training.tolist(), unseen.tolist(), higher_num_subgoals

def get_instructions_tokenizer(instructions: NaturalLanguageInstruction, padding_len=None) -> InstructionTokenizer:
    return InstructionTokenizer([instr[0] for instr in instructions], padding_len)
    