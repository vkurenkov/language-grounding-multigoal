# Reinforcement Learning: Sequential goals with arbitrary order
The aim of the repository is to find out how reinforcement learning techniques can be applied to find a solution for a given instruction with one important property: the instruction contains sub-goals with specific (and possibly arbitrary) order. We investigate whether the agent is able to generalize to unseen order within the instruction and objects placement. Also, we examine a possibility to feed this kind of instructions directly into the agent, i.e. raw natural language.

Basically, the problem we aim to solve is based on two capstones: solve a sequence of sub-goals in the right order and build the sequence from the natural language instruction. In the end, we would like to build an end-to-end architecture. But for now, it is an open question, whether it is feasible or we should extract sub-goals explicitly.

For now, let us focus on the first issue, solving a sequence of sub-goals.
We aim to solve the problem in the modified GridWorld environment.

## Environment

Environments can be of different difficulty, in this case, we list few important properties:
  - Observation
    - Provides a list of items the agent already visited
    - Provides a full state of the map right now
  - Reward Shaping
    - Reward as a distance metric to the current target item
    - A reward for achieving any target item along the right trajectory
    - A reward is only given at the end of the right trajectory
    
Some of the properties above lead to the POMDP-environment which is hard. Other properties incur sparsity, and it is still a crucial and ongoing challenge for the reinforcement learning field.

## Research Questions
- How good is the agent in an MDP-environment?
  - Training performance metrics: the total number of timesteps; average reward; an average number of timesteps;
  - Generalization performance metrics: success rate; an average number of timesteps;
  - Can it solve the multitask problems?
  
- How good is the agent in a POMDP-environment?
  - Performance metrics: the total number of timesteps; average reward; success rate;
  * Can it solve the multitask problems?  
  
- Is it possible to create encodings for such instructions automatically from natural language?
  - E.g "Find the triangle, then the circle"; "Find the triangle but first the circle"
  - Can the agent use relevant information?
    - E.g "Find the triangle in the red zone and the find the circle in the green zone"
