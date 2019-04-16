"""
This file contains evaluation protocol for the agent.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F

from utils.training import fix_random_seeds
from envs.goal.env  import GoalStatus
from instructions	import get_instructions
from instructions   import get_instructions_tokenizer
from collections    import deque, defaultdict

# Computational model
from experiments.dueling_dqn_ga_per.model 	   import Model
from experiments.dueling_dqn_ga_per.model      import prepare_model_input
from experiments.dueling_dqn_ga_per.parameters import device

# Prioritized eperience replay
from experiments.dueling_dqn_ga_per.per        import PrioritizedProportionalReplay


def benchmark_one(name, instructions, layouts, model, stack_frames, tokenizer, env_definition, max_episode_len, gamma):
    total_successes = []
    total_lengths   = []
    total_rewards   = []
    result          = defaultdict(list)
    for ind, layout in enumerate(layouts):
        for instruction in instructions:
            instruction_raw   = instruction[0]
            instruction_items = instruction[1]

            # Instruction encoding
            instruction_idx = tokenizer.text_to_ids(instruction_raw)
            instruction_idx = np.array(instruction_idx)
            instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

            # Environment reset
            env = env_definition.build_env(instruction_items)
            env.fix_initial_positions(layout)
            observation, reward, done, _ = env.reset()

            # Keep last frames
            last_frames = deque([observation for _ in range(stack_frames)], stack_frames)

            num_steps = 0
            rewards   = []
            while num_steps < max_episode_len and env.goal_status() == GoalStatus.IN_PROGRESS and not done:
                # Action inference
                with torch.no_grad():
                    model_input 	 = prepare_model_input(list(last_frames), instruction_idx)
                    values 			 = model(model_input)
                
                # Greedy
                action = torch.argmax(values, dim=-1).cpu().item()

                observation, reward, done,  _ = env.step(action)
                # Check if it was the last step and we could not successfuly eecute the instruction
                if num_steps >= max_episode_len and done and reward < 10:
                    reward += -10

                last_frames.append(observation)
                rewards.append(reward)
                num_steps += 1

            success = 0
            if env.goal_status() == GoalStatus.SUCCESS:
                success = 1
            elif env.goal_status() == GoalStatus.FAILURE or env.goal_status() == GoalStatus.IN_PROGRESS:
                success = 0
                num_steps = max_episode_len
            
            total_successes.append(success)
            total_lengths.append(num_steps)

            # Calculate discounted reward (what you were optimizing for)
            discounted_reward = 0.0
            for reward in reversed(rewards):
                discounted_reward = reward + discounted_reward * gamma
            total_rewards.append(discounted_reward)

            result[instruction_raw].append({"reward": discounted_reward, "success": success, "steps": num_steps})

    print("{} (success rate): {}%".format(name, np.mean(total_successes)))
    print("{} (trajectory len): {}".format(name, np.mean(total_lengths)))
    print("{} (reward): {}".format(name, np.mean(total_rewards)))
    print()
    return result

def benchmark_all():
    # Experimental environment and layouts
    from experiments.dueling_dqn_ga_per.parameters import env_definition
    from experiments.dueling_dqn_ga_per.parameters import layouts_parameters

    # Experimental instructions
    from experiments.dueling_dqn_ga_per.parameters import instructions_parameters

    # Agent's training and testing parameters
    from experiments.dueling_dqn_ga_per.parameters import train_parameters
    from experiments.dueling_dqn_ga_per.parameters import test_parameters
    from experiments.dueling_dqn_ga_per.parameters import get_experiment_folder


    # Experimental folder
    experiment_folder        = get_experiment_folder()

    # Experimental instructions
    train_instructions, unseen_instructions, higher_subgoals_instructions = get_instructions(
                                                                                instructions_parameters["level"], 
                                                                                instructions_parameters["max_train_subgoals"], 
                                                                                instructions_parameters["unseen_proportion"],
                                                                                instructions_parameters["seed"])
    tokenizer         		 = get_instructions_tokenizer(
                                train_instructions,
                                train_parameters["padding_len"])

    # Experimental layouts
    layouts                  = env_definition.build_env().generate_layouts(
                                layouts_parameters["num_train"] + layouts_parameters["num_test"],
                                layouts_parameters["seed"])
    train_layouts			 = layouts[:layouts_parameters["num_train"]]
    test_layouts             = layouts[layouts_parameters["num_train"]:]

    stack_frames = train_parameters["stack_frames"]
    model = Model(tokenizer.get_vocabulary_size(), stack_frames, train_parameters["max_episode_len"])
    model.load_state_dict(torch.load(os.path.join(experiment_folder, "best.model"), map_location='cpu'))
    model.to(device)
    model.eval()

    eval_pairs = {
        "Instructions seen, layouts seen": (train_instructions, train_layouts),
        "Instructions seen, layouts unseen": (train_instructions, test_layouts),

        "Instructions unseen, layouts seen": (unseen_instructions, train_layouts),
        "Instructions unseen, layouts unseen": (unseen_instructions, test_layouts),

        "Instructions high, layouts seen": (higher_subgoals_instructions, train_layouts),
        "Instructions high, layouts unseen": (higher_subgoals_instructions, test_layouts)
    }
    result     = {}

    for name in eval_pairs:
        instructions = eval_pairs[name][0]
        layouts      = eval_pairs[name][1]

        result[name] = benchmark_one(name, instructions, layouts, model,
                        stack_frames, tokenizer, env_definition, 
                        train_parameters["max_episode_len"], train_parameters["gamma"])

    return result
