instruction = ["Triangle", "Square", "Circle"]
vocabulary = set(instruction)

#######################################
# The essense of the research is to etract sub-goals from the instruction based on the reward signal
# And to achieve these sub-goals in the proper order!
# Sub-goals can be structured, e.g. encoded binary
# Or they can be unstructured, e.g. in natural language

#######################################
# What about novelty?
    # 1 - Eecute a sequence of action (if compared to language grounding)
        # Argument against: Visual semantic planning; Hierarchical and Interpretable Skill-Acquisition
            # These papers about multi-task learning, sub-goals are deduced from the reward signal only
            # But we operate on the level, where sub-goals are eplicitly given in encoded view or as a natural language
            # This encoding or raw format contains information about the proper order of the eecution
            # The question is how to get juice from this information so the agent can generalize to unseen orders and object placements
    # 2 - Etract sub-goals automatically from the natural language instruction (reward sparsity)
        # Natural Language instructions are hard, e.g. "Find triangle but find circle first"
        # They can be of any length (I am not sure, we should address it in the research)

#######################################
# Partial view DOES NOT PROVIDE AN AGENT'S POSITION
# Ideally, we do not where we are in the world space!!
# Only: okay, i am one meter away from the wall

#######################################
# With partial view, the task is about eploration of the environment in the proper way
# To understand, that I have been here before and to move to another direction
# This is intrinsic for our environment, is it true for others?
# If this is intrinsic for our environment, we can abuse this property
# And based on this abusing - solve the environment, is it possible? Is it okay?

#######################################
# Find all papers that try to solve sub-goals
# Probably, it can be a life-long learning?
# Multi-task learning?

#######################################
# From structured data to unstructured data
    # Structured: e.g. a binary encoding
    # Unstructured: natural language
# Solving for structured data is simpler
# Solving for structured data does not provide guarantees for solving for unstructured data
# Solving for unstructured data does not provide guarantees for solivng for structured data

#######################################
# What arranges should we train, so the agent can generalize well?
# Generalization to longer sequences?

#######################################
# What about interpretability?
    # The agent should spit out what is he doing right now
    # I.e. Hierarchical Skill-Acquisition paper
    # Or spit-out embedding vectors?

#######################################
# What about extension to multi-task instructions?
    # E.g. "Find a cabinet and open it then take an apple from there"
    # E.g. "Find a fridge and take an apple" (Should the agent deduce that the apple in the fridge?)

#######################################
# What about temporal instructions?
    # E.g. "Find a triangle and after 40-timesteps reach the circle."

#######################################
# Ideas
    # 1 - What about to use current lifetime step in state's encoding?
    # 2 - What about to use previous reward in state's encoding?
        # This is not as good as I would like it to be
        # Since it eplicitly abuses reward shaping
        # It is not going to work in the case of the sparse rewards

#######################################
# Cases
    # 1 - Environment provides which items we have visited
        # Train UVFA or Hordes to find different items 
        # -> Switch sub-goals as long as you have solved the current one
    # 2 - Environment does not provide which items we have visited
        # (!) Somehow need to find out which sub-goals were already solved
        
        # 1
        # Can use trainig signal for easier sparse reward - that gives you a cookie for achieving one item
        # This signal can be used to connect states and sub-goals

        # 2
        # What to do in the case of completely sparse rewards in the environment?
        # How to find out sub-goals were solved? This is why we use RL and Bellman Equation
        # Curriculum learning for help? 
            # Kind of an initialization for the harder problem
            # Train for finding one item, then another, then both of them, them one more and so on
