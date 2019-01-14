import logging
import random
import torch
import gym

import torch.optim            as optim
import torch.nn               as nn
import torch.nn.functional    as F
import torch.multiprocessing  as mp
import numpy                  as np

from agents.chaplot.model                          import A3C_LSTM_GA
from agents.agent                                  import Agent
from torch.autograd                                import Variable
from envs.definitions                              import InstructionEnvironmentDefinition
from envs.definitions                              import NaturalLanguageInstruction
from envs.definitions                              import Instruction
from envs.gridworld_simple.instructions.tokenizer  import InstructionTokenizer
from utils.training                                import fix_random_seeds
from typing                                        import List, Tuple
from typing                                        import Optional, Dict
from tensorboardX                                  import SummaryWriter



def ensure_shared_grads(model: nn.Module, shared_model: nn.Module):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def sample_instruction(instructions: List[NaturalLanguageInstruction]) -> NaturalLanguageInstruction:
    return instructions[random.randint(0, len(instructions) - 1)]

def train(
    env_definition:      InstructionEnvironmentDefinition,
    shared_model:        A3C_LSTM_GA,
    instructions:        List[NaturalLanguageInstruction],
    tokenizer:           InstructionTokenizer,
    logdir:              str,
    seed:                int, 
    input_size:          int,
    max_episode_len:     int,
    learning_rate:       int,
    gamma:               int,
    tau:                 int,
    num_bootstrap_steps: int):

    # Agents should start from different seeds
    # Otherwise it will lead to the same experience (not desirable at all)
    fix_random_seeds(seed)

    logger = SummaryWriter(logdir)

    model = A3C_LSTM_GA(input_size, max_episode_len)
    model.train()
    optimizer = optim.SGD(shared_model.parameters(), lr=learning_rate)


    instruction     = sample_instruction(instructions)
    env             = env_definition.build_env(instruction[1])


    observation, _, _, _ = env.reset()
    observation          = torch.from_numpy(observation).float()/255.0
    instruction_idx      = tokenizer.text_to_ids(instruction[0])
    instruction_idx      = np.array(instruction_idx)
    instruction_idx      = torch.from_numpy(instruction_idx).view(1, -1)


    done            = True
    episode_length  = 0
    episode_rewards = []
    total_steps     = 0
    total_episodes  = 0

    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        if done:
            episode_length  = 0
            episode_rewards = []
            cx = torch.zeros(1, 256, requires_grad=True)
            hx = torch.zeros(1, 256, requires_grad=True)
        else:
            cx = cx.clone().detach().requires_grad_(True)
            hx = hx.clone().detach().requires_grad_(True)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(num_bootstrap_steps):
            episode_length += 1
            total_steps      += 1

            tx = torch.tensor(np.array([episode_length]), dtype=torch.int64)
            
            value, logit, (hx, cx) = model((torch.tensor(observation).unsqueeze(0),
                                            torch.tensor(instruction_idx),
                                            (tx, hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, torch.tensor(action))

            action = action.numpy()[0, 0]
            observation, reward, done,  _ = env.step(action)

            done = done or episode_length >= max_episode_len

            if done:
                observation, _, _, _ = env.reset()
                observation          = torch.from_numpy(observation).float()/255.0
                instruction_idx      = tokenizer.text_to_ids(instruction[0])
                instruction_idx      = np.array(instruction_idx)
                instruction_idx      = torch.from_numpy(instruction_idx).view(1, -1)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            tx = torch.tensor(np.array([episode_length]), dtype=torch.int64)
            value, _, _ = model((torch.from_numpy(observation).unsqueeze(0),
                                 instruction_idx.clone().detach(), (tx, hx, cx)))
            R = value.data

        values.append(torch.tensor(R))
        policy_loss = 0
        value_loss = 0
        R = torch.tensor(R)

        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + gamma * \
                values[i + 1].data - values[i].data
            gae = gae * gamma * tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * torch.tensor(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        logger.add_scalar("Rollout/GAE Reward", gae.data[0, 0], total_steps)
        logger.add_scalar("Rollout/Policy loss", policy_loss.data[0, 0], total_steps)
        logger.add_scalar("Rollout/Value loss", value_loss.data[0, 0], total_steps)
        logger.add_scalar("Rollout/Total loss", policy_loss.data[0, 0] + 0.5 * value_loss.data[0, 0], total_steps)

        print(total_steps)
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        if done:
            logger.add_scalar("Episode/Reward (sum)", np.sum(episode_rewards), total_episodes)
            logger.add_scalar("Episode/Length", episode_length, total_episodes)
            total_episodes += 1


class GatedAttentionAgent(Agent):
    def __init__(self, instruction_tokenizer: InstructionTokenizer,
                       gamma=0.99, tau=1.00, num_bootstrap_steps=20,
                       learning_rate=0.001, num_processes=4, max_episode_len=100,
                       seed=0):
        super(Agent, self).__init__()

        self._gamma                 = gamma
        self._tau                   = tau
        self._num_bootstrap_steps   = num_bootstrap_steps
        self._num_processes         = num_processes
        self._learning_rate         = learning_rate
        self._seed                  = seed
        self._max_episode_len       = max_episode_len
        self._instruction_tokenizer = instruction_tokenizer

    def train_init(self, 
        env_definition: InstructionEnvironmentDefinition, 
        training_instructions: List[NaturalLanguageInstruction]) -> None:

        shared_model = A3C_LSTM_GA(
                            self._instruction_tokenizer.get_vocabulary_size(),
                            self._max_episode_len)
        shared_model.share_memory()

        # train(
        #     env_definition,
        #     shared_model,
        #     training_instructions,
        #     self._instruction_tokenizer,
        #     self._log_writer,
        #     self._seed + 0,
        #     self._instruction_tokenizer.get_vocabulary_size(),
        #     self._max_episode_len,
        #     self._learning_rate,
        #     self._gamma,
        #     self._tau,
        #     self._num_bootstrap_steps
        # )

        processes = []

        # Start the training thread(s)
        for rank in range(0, self._num_processes):
            args = (
                env_definition,
                shared_model,
                training_instructions,
                self._instruction_tokenizer,
                self._log_writer.log_dir + "/agent_{}".format(rank),
                self._seed,
                self._instruction_tokenizer.get_vocabulary_size(),
                self._max_episode_len,
                self._learning_rate,
                self._gamma,
                self._tau,
                self._num_bootstrap_steps
            )

            p = mp.Process(target=train, args=args)
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()

    def train_step(self) -> None:
        raise NotImplementedError()

    def train_num_steps(self) -> int:
        raise NotImplementedError()

    def train_is_done(self) -> bool:
        return True
    
    def reset(self) -> None:
        """
        Should reset to the initial state for an episode.
        """
        raise NotImplementedError()

    def act(self, observation, instruction: Instruction, env: Optional[gym.Env] = None) -> Optional[int]:
        raise NotImplementedError()

    def parameters(self) -> Dict:
        return {
            "tau": self._tau,
            "gamma": self._gamma,
            "num_bootstrap_steps": self._num_bootstrap_steps,
            "num_processes": self._num_processes,
            "seed": self._seed
        }

    def name(self) -> str:
        return "gated-attention-a3c-lstm-seed_{}-num_proc_{}-num_steps_{}-tau_{}-gamma_{}" \
                .format(self._seed, self._num_processes, self._num_bootstrap_steps,
                        self._tau, self._gamma)