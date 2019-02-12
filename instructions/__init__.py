import json
import os

from envs.definitions                              import NaturalLanguageInstruction
from envs.definitions                              import Instruction
from envs.gridworld_simple.instructions.tokenizer  import InstructionTokenizer
from typing                                        import List, Tuple

def get_this_file_path() -> str:
    return os.path.dirname(__file__)

def get_instructions(path: str) -> List[NaturalLanguageInstruction]:
    dataset_path = os.path.join(path)
    with open(dataset_path, mode="r") as f:
        dataset = json.load(f)

    return [(instruction["raw"], instruction["objects_real_order"]) for instruction in dataset["instructions"]]

def get_level1_instructions() -> List[NaturalLanguageInstruction]:
    dataset_path = os.path.join(get_this_file_path(), "level1.json")
    return get_instructions(dataset_path)

def get_level0_instructions() -> List[NaturalLanguageInstruction]:
    dataset_path = os.path.join(get_this_file_path(), "level0.json")
    return get_instructions(dataset_path)

def get_instructions_tokenizer(instructions: NaturalLanguageInstruction, padding_len=None) -> InstructionTokenizer:
    return InstructionTokenizer([instr[0] for instr in instructions], padding_len)