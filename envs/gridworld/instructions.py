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
name_mapping = {
    "red": 0,
    "blue": 1,
    "green": 2,
    "yellow": 3,
    "black": 4
}


def generate_compound_instructions(objects, max_instruction_length):
    '''
    :param objects: A dictionary of object names and their corresponding ids.
    :param max_instruction_length: Maximum length of any generated instruction (in terms of sub-goals).

    :return compounds: A list of generated instructions in natural language and object ids in the right order.
    '''
    names = list(objects.keys())

    possible_compounds = traverse_fsm(states, transitions, "start", [], depth=0, max_depth=max_instruction_length)
    compounds = []
    for seq in possible_compounds:
        permutations = itertools.permutations(names, len(seq))
        for permutation in permutations:
            instruction = ""
            instruction_order = []
            conjunctions = []
            for ind, (state, name) in enumerate(zip(seq, permutation)):
                instruction += states[state].format(name)
                instruction_order.insert(states_order[state](ind), objects[name])
                conjunctions.append(conjunction_from_state(states[state]))

            compounds.append({"raw": instruction, "ordered_objects": instruction_order, "conjunctions": conjunctions})
    
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


def generate_train_test_split(instructions_dataset, train_split=0.8):
    '''
    params:
        instructions_dataset - {conjunctions: [], name_ids: [], instructions: []}
    returns:
        train_instructions - []
        test_instructions - []

    Randomly generates constrained train/test split.
    Constraints:
        - All conjunctions must be present in the training and testing sets
        - All objects must be present in the training and testing sets
    '''
    conjunctions = instructions_dataset["conjunctions"]
    name_ids = instructions_dataset["name_ids"]

    def has_all_conjunctions(instructions):
        left_conjunctions = set(conjunctions)
        for instruction in instructions:
            for conjunction in instruction["conjunctions"]:
                if conjunction in left_conjunctions:
                    left_conjunctions.remove(conjunction)
                if len(left_conjunctions) == 0:
                    return True

        return False

    def has_all_objects(instructions):
        left_ids = set(name_ids.values())
        for instruction in instructions:
            for obj_id in instruction["ordered_objects"]:
                if obj_id in left_ids:
                    left_ids.remove(obj_id)
                if len(left_ids) == 0:
                    return True

        return False

    def split(instructions):
        # Shuffles in place
        random.shuffle(instructions)
        num_instructions = len(instructions)
        train_instructions = instructions[:int(num_instructions * train_split)]
        test_instructions = instructions[int(num_instructions * train_split):]

        return train_instructions, test_instructions

    # Try until all constraints are met
    train_instructions, test_instructions = split(instructions_dataset["instructions"])
    while (not has_all_conjunctions(train_instructions)) or (not has_all_objects(train_instructions)) \
        or (not has_all_conjunctions(test_instructions)) or (not has_all_objects(test_instructions)):
        train_instructions, test_instructions = split(instructions_dataset["instructions"])

    return train_instructions, test_instructions


def conjunction_from_state(state):
    return str.strip(state.format(""))


if __name__ == "__main__":
    instructions = generate_compound_instructions(name_mapping, max_instruction_length=2)
    with open("instructions.json", mode="w", encoding="utf-8") as f:
        # Save instructions
        instructions_dataset = {
            "conjunctions": [conjunction_from_state(state) for state in states.values()],
            "name_ids": name_mapping,
            "instructions": instructions
        }
        json.dump(instructions_dataset, f, indent=4, separators=(",", ":"))

        # Generate train/test split
        train, test = generate_train_test_split(instructions_dataset)
        with open("instructions_train.json", mode="w", encoding="utf-8") as f:
            json.dump(train, f, indent=4, separators=(",", ":"))
        with open("instructions_test.json", mode="w", encoding="utf-8") as f:
            json.dump(test, f, indent=4, separators=(",", ":"))