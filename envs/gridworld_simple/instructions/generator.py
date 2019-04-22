'''
Generate natural language instructions and provides a sequence of goals to be completed.
'''
import random
import itertools
import json
from copy import copy


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
    #"start": ["and then", "after that", "but first", "comma", "but before"], # All instructions
    #"start": ["comma", "but first"],                                         # Level2 - Only comma
    #"start": ["comma", "but before"],                                        # Level3 - Only comma
    "start": ["comma", "but before", "but first"],                            # Level4 - Only comma

    #"and then": ["after that", "but first", "but before"],                   # All instructions
    # "after that": ["after that", "and then", "but first", "but before"],    # All instructions
    "but first": [],                                                          # All instructions

    #"comma": ["after that", "and then", "but first", "comma", "but before"], # All instructions
    #"comma": ["comma", "but first"],                                         # Level2 - Only comma
    #"comma": ["comma", "but before"],                                        # Level3 - Only comma
    "comma": ["comma", "but before", "but first"],                            # Level4 - Only comma

    #"but before": ["after that", "and then", "but first"]                    # All instructions
    #"but before": []                                                         # Level3 - Only comma
    "but before": ["but first"]                                               # Level4 - Only comma
}
states_order = {
    "start": lambda t : 0,
    "and then": lambda t : t,
    "after that": lambda t: t,
    "but first": lambda t: 0,
    "comma": lambda t: t,
    "but before": lambda t: t - 1
}
name_mapping = {
    "red": 0,
    "blue": 1,
    "green": 2
}


def generate_compound_instructions(objects, max_instruction_length):
    '''
    :param objects: A dictionary of object names and their corresponding ids.
    :param max_instruction_length: Maximum length of any generated instruction (in terms of sub-goals).

    :return compounds: A list of generated instructions in natural language and object ids in the right order.
    '''
    def no_from_self_to_itself(objects):
        for ind in range(len(objects) - 1):
            if objects[ind] == objects[ind + 1]:
                return False
        return True
    names = list(objects.keys())

    total_possibilities = 0
    valid_possibilities = 0

    possible_compounds = traverse_fsm(states, transitions, "start", [], depth=0, max_depth=max_instruction_length)
    compounds = []
    for seq in possible_compounds:
        permutations = itertools.product(names, repeat=len(seq))
        for permutation in permutations:
            instruction               = ""
            instruction_real_order         = []
            instruction_text_order = []
            conjunctions              = []
            for ind, (state, name) in enumerate(zip(seq, permutation)):
                instruction += states[state].format(name)
                instruction_real_order.insert(states_order[state](ind), objects[name])
                instruction_text_order.append(objects[name])
                conjunctions.append(conjunction_from_state(states[state]))

            # Instructions that require to move from "red" to "red" and etc. are ignore
            total_possibilities += 1
            if no_from_self_to_itself(instruction_real_order):
                compounds.append({
                    "raw": instruction, 
                    "objects_real_order": instruction_real_order,
                    "objects_text_order": instruction_text_order,
                    "conjunctions": conjunctions
                })
                valid_possibilities += 1
    
    print(valid_possibilities)
    print(total_possibilities)
    print(valid_possibilities / total_possibilities)
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


def conjunction_from_state(state):
    return str.strip(state.format(""))


def extract_level1_instructions(instructions):
    def is_the_same_order(instruction):
        return instruction["objects_real_order"] == instruction["objects_text_order"]

    return list(filter(is_the_same_order, instructions))


if __name__ == "__main__":
    instructions         = generate_compound_instructions(name_mapping, max_instruction_length=6)
    #instructions_level1  = extract_level1_instructions(instructions)
    instructions_dataset = {
        "conjunctions": [conjunction_from_state(state) for state in states.values()],
        "name_ids": name_mapping,
    }

    # with open("instructions.json", mode="w", encoding="utf-8") as f:
    #     instructions_dataset["instructions"] = instructions
    #     json.dump(instructions_dataset, f, indent=4, separators=(",", ":"))
    with open("instructions-level4.json", mode="w", encoding="utf-8") as f:
        instructions_dataset["instructions"] = instructions
        json.dump(instructions_dataset, f, indent=4, separators=(",", ":"))