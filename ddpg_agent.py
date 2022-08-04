import numpy as np

from model import Actor, Critic

import torch
import torch.optim as optim

WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, state_size, full_state_size, action_size, full_action_size, lr_actor, lr_critic, actor_l1, actor_l2, critic_l1, critic_l2, noise_scalar=0.25, noise_decay=0.9999):
        self.state_size = state_size
        self.full_state_size = full_state_size
        self.action_size = action_size
        self.full_action_size = full_action_size
        self.noise_scalar = noise_scalar
        self.noise_decay = noise_decay

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, fc1_units=actor_l1, fc2_units=actor_l2).to(device)
        self.actor_target = Actor(state_size, action_size, fc1_units=actor_l1, fc2_units=actor_l2).to(device)
        self.actor_target.eval()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(full_state_size, full_action_size, fcs1_units=critic_l1, fc2_units=critic_l2).to(device)
        self.critic_target = Critic(full_state_size, full_action_size, fcs1_units=critic_l1, fc2_units=critic_l2).to(device)
        self.critic_target.eval()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        # make sure that at the beginning local and target networks are the same
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy with optional noise."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += np.random.randn(self.action_size) * self.noise_scalar
            self.noise_scalar *= self.noise_decay
        return np.clip(action, -1, 1)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
