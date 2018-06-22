# Reinforcement Learning: Sequential goals with arbitrary order
The aim of the repository is to find out how reinforcement learning techniques can be applied to find a solution for a given instruction with one important property: the instruction contains sub-goals with specific (and possibly arbitrary) order. We investigate whether the agent is able to generalize to unseen order within the instruction and objects placement. Also, we examine a possibility to feed this kind of instructions directly into the agent, i.e. raw natural language.

For the simplicity, we do not examine instructions with more than 3 sub-goals and leave it as a future work. 
We aim to solve the problem in the modified GridWorld environment.

## Environment

## Research Questions
- How good is the agent in an MDP-environment?
  - Training performance metrics: the total number of timesteps; average reward; average number of timesteps;
  - Generalization performance metrics: success rate; average number of timesteps;
  - Can it solve the multi-task problems?
  
- How good is the agent in an POMDP-environment?
  - Performance metrics: the total number of timesteps; average reward; success rate;
  * Can it solve the multi-task problems?  
  
- Is it possible to create encodings for such instructions automatically from natural language?
  - E.g "Find the triangle, then the circle"; "Find the triangle but first the circle"
  - Can the agent use relevant information?
    - E.g "Find the triangle in the red zone and the find the circle in the green zone"
