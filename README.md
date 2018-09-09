# Execution of the complex natural language instructions in virtual environments

# Perspective 1.0

### Problem statement
  - Given
    - A corpus of textual instructions, where an instruction has following properties
        - Possibly contains sub-instructions (*Go to the red and then to the green but to the blue first*)
        - Possibly contains conditional sub-instructions (*Go to the red if there is a green near by*)
        - Possibly contains irrelevant information (*Hey, Irene. Please, go to the red!*)
        - Possibly contains useful relevant information (*Go to the table in the kitchen*)
    - A signal for successful completion one of the subtasks in the given instruction
  - Output
    - A trajectory that resolves the instruction
    - An explanation of what the agent does at every timestep
    - Generalization to unseen instructions
    - Generalization to unseen mutations of the given environment
    
### Goals
  - Provide a qualitative analysis on execution of compound and conditional intstructions
  - Make an architecture that is able to describe what the agent is doing in natural language (how to measure?)
    
### Metrics
  - Sample complexity
  - Average reward
  - Success rate

# Perspective 0.0

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

Based on the properties above, we start from the "easiest" environments and proceed to the "hardest" ones.

***
**First level**: Provide as much information as possible but vary sparsity of the reward.
  1. List of visited items + full view of the map + reward as a distance metric
  2. List of visited items + full view of the map + reward at every item
  3. ~~List of visited items + full view of the map + reward at the last item~~ (Does not make any sense, since we deduce visited items at the training time)
  
***  
**Second level**: The same as above but with the partial view of the map.
  1. List of visited items + partial view of the map + reward as a distance metric
  2. List of visited items + partial view of the map + reward at every item
  3. ~~List of visited items + partial view of the map + reward at the last item~~ (Does not make any sense, since we deduce visited items at the training time)
  
***  
**Third level**: Do not deduce the list of visited items manually and vary the sparsity of the reward.
  1. Full view of the map + reward as a distance metric
  2. Full view of the map + reward at every item
  3. Full view of the map + reward at the last item
  
***  
**Fourth level**: The same as above but with the partial view of the map.
  1. Partial view of the map + reward as a distance metric
  2. Partial view of the map + reward at every item
  3. Partial view of the map + reward at the last item
 
***

Having a list of visited items gives us an opportunity to use several value functions (e.g. Horde) or Universal Value Function Approximator approach. So it is relatively easy to solve such environment when compared to not having a list of visited items. The agent should somehow find ways to infer this list. And reward sparsity takes a crucial role in this case, as we will see later.

Partial view + distance to the current item - is a POMDP in cases where equidistant items are not visible to the agent.
We artificially create a limitation, such that we do not deduce the list of visited items based on the view of the map. But, one of the interesting outcomes would be to have an agent that can deduce such information automatically.

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
