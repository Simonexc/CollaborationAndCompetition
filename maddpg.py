import numpy as np
import random
from collections import namedtuple, deque


import torch
import torch.nn.functional as F

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, agents, learn_every_t_steps, learn_n_times):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.learn_every_t_steps = learn_every_t_steps
        self.learn_n_times = learn_n_times

        self.agents = agents

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        self.counter = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        self.counter += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.counter % self.learn_every_t_steps == 0:
            for _ in range(self.learn_n_times):
                experiences = self.memory.sample()
                for agent_num in range(len(self.agents)):
                    self.learn(experiences, GAMMA, agent_num)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        return np.array([agent.act(state, add_noise) for agent, state in zip(self.agents, states)])

    def target_act(self, states):
        full_action_next = [agent.actor_target(state) for agent, state in zip(self.agents, states)]
        return torch.cat(full_action_next, dim=1)

    def reformat_data(self, tensor):
        if len(tensor.shape) == 3:
            return torch.stack([tensor[:, i, :] for i in range(len(self.agents))]).to(device)
        return torch.stack([tensor[:, i] for i in range(len(self.agents))]).to(device)

    def learn(self, experiences, gamma, agent_num):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        full_states = states.reshape(BATCH_SIZE, -1).to(device)
        full_next_states = next_states.reshape(BATCH_SIZE, -1).to(device)
        full_actions = actions.reshape(BATCH_SIZE, -1).to(device)

        states = self.reformat_data(states)
        rewards = self.reformat_data(rewards)
        next_states = self.reformat_data(next_states)
        dones = self.reformat_data(dones)

        reward = rewards[agent_num].reshape(-1, 1)
        done = dones[agent_num].reshape(-1, 1)
        agent = self.agents[agent_num]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            full_actions_next = self.target_act(next_states)
            Q_targets_next = agent.critic_target(full_next_states, full_actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = reward + (gamma * Q_targets_next * (1 - done))
        # Compute critic loss
        Q_expected = agent.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        full_actions_prediction = []
        for i, (current_agent, state) in enumerate(zip(self.agents, states)):
            action = current_agent.actor_local(state)
            if i != agent_num:
                action = action.detach()
            full_actions_prediction.append(action)
        full_actions_prediction = torch.cat(full_actions_prediction, dim=1)

        actor_loss = -agent.critic_local(full_states, full_actions_prediction).mean()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis=0)).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None], axis=0)).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None], axis=0)).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None], axis=0)).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)