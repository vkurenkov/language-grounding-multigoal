'''
Generate natural language instructions and provides a sequence of goals to be completed.
'''
import random
import itertools
import json
from copy import copy


def generate_compound_instructions(objects, max_instruction_length):
    '''
    :param objects: A dictionary of object names and their corresponding ids.
    :param max_instruction_length: Maximum length of any generated instruction (in terms of sub-goals).

    :return compounds: A list of generated instructions in natural language and object ids in the right order.
    '''
    names = list(objects.keys())

    # FSM for instruction generation and how to order them
    states = {
        "start": "Go to the {}",
        "and then": " and then go to the {}",
        "after that": " and after that go to the {}",
        "but first": " but first go to the {}",
        "comma": ", go to the {}",
        "but before": " but before go to the {}"
    }
    transitions = {
        "start": ["and then", "after that", "but first", "comma", "but before"],
        "and then": ["after that", "but first", "but before"],
        "after that": ["after that", "and then", "but first", "but before"],
        "but first": [],
        "comma": ["after that", "and then", "but first", "comma", "but before"],
        "but before": ["after that", "and then", "but first"]
    }
    states_order = {
        "start": lambda t : 0,
        "and then": lambda t : t,
        "after that": lambda t: t,
        "but first": lambda t: 0,
        "comma": lambda t: t,
        "but before": lambda t: t - 1
    }

    possible_compounds = traverse_fsm(states, transitions, "start", [], depth=0, max_depth=max_instruction_length)
    compounds = []
    for seq in possible_compounds:
        permutations = itertools.permutations(names, len(seq))
        for permutation in permutations:
            instruction = ""
            instruction_order = []
            for ind, (state, name) in enumerate(zip(seq, permutation)):
                instruction += states[state].format(name)
                instruction_order.insert(states_order[state](ind), objects[name])
            compounds.append((instruction, instruction_order))
    
    return compounds

def traverse_fsm(states, transitions, cur_state, cur_states, depth, max_depth):
    if depth == max_depth:
        return []

    cur_states_ = copy(cur_states)
    cur_states_.append(cur_state)
    total_states = [cur_states_]
    for transition in transitions[cur_state]:
        consecutive_states = traverse_fsm(states, transitions, transition, cur_states_, depth + 1, max_depth)
        for states in consecutive_states:
            total_states.append(states)

    return total_states

if __name__ == "__main__":
    name_mapping = {
        "red": 0,
        "blue": 1,
        "green": 2,
        "yellow": 3,
        "black": 4
    }
    instructions = generate_compound_instructions(name_mapping, max_instruction_length=5)
    with open("instructions.json", mode="w", encoding="utf-8") as f:
        json.dump({
            "name_ids": name_mapping,
            "instructions": instructions
        }, f, indent=4, separators=(",", ":"))