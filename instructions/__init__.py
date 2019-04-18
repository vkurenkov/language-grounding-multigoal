import json
import os
import random
import numpy as np

from envs.definitions                              import NaturalLanguageInstruction
from envs.definitions                              import Instruction
from envs.gridworld_simple.instructions.tokenizer  import InstructionTokenizer
from typing                                        import List, Tuple
from utils.training                                import fix_random_seeds
from collections                                   import defaultdict

def get_this_file_path() -> str:
    return os.path.dirname(__file__)

def read_instructions_from_disk(path: str) -> List[NaturalLanguageInstruction]:
    dataset_path = os.path.join(path)
    with open(dataset_path, mode="r") as f:
        dataset = json.load(f)

    return [(instruction["raw"], instruction["objects_real_order"]) for instruction in dataset["instructions"]]

def get_instructions(level: int, max_train_subgoals: int, unseen_proportion: float, seed: int, conjunctions: str):
    """
    Divides the specified level dataset into 3 parts: training instructions, unseen combinations, and higher number of subgoals.

    :param max_train_subgoals: Maximum number of available sub-goals during training. Others will go to the 3rd part. 
    :param unseen_proportion: Split between training instructions and unseen combinations.
    :returns: training instructions, unseen combinations, higher number of sub-goals
    """
    # We want our splits to be reproducible
    fix_random_seeds(seed)

    dataset_path = os.path.join(get_this_file_path(), "{}/level{}.json".format(conjunctions, level))
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

    # Group instruction by their item visitation order
    # We split seen and unseen instructions based on the items to visit not on thei conjunctions
    train_instructions = defaultdict(list)
    for instruction in training:
        train_instructions[tuple(instruction[1])].append(instruction)

    num_unique_combinations = len(train_instructions.keys())
    num_unseen_combinations = int(num_unique_combinations * unseen_proportion)
    unseen_indices          = set(np.random.choice(range(num_unique_combinations), size=num_unseen_combinations, replace=False))
    train  = []
    unseen = []
    for ind, key in enumerate(sorted(train_instructions.keys())):
        if ind in unseen_indices:
            unseen.extend(train_instructions[key])
        else:
            train.extend(train_instructions[key])

    print("Level{}: Training {} instructions.".format(level, len(train)))
    print("Level{}: Unseen {} instructions.".format(level, len(unseen)))
    print("Level{}: Higher subgoals {} instructions.".format(level, len(higher_num_subgoals)))

    return train, unseen, higher_num_subgoals

def get_instructions_tokenizer(instructions: NaturalLanguageInstruction, padding_len=None) -> InstructionTokenizer:
    return InstructionTokenizer([instr[0] for instr in instructions], padding_len)
    