import random
from collections import namedtuple, deque
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from replay_buffer import ReplayBuffer
from default_hyperparameters import SEED, BUFFER_SIZE, BATCH_SIZE, START_SINCE,\
                                    GAMMA, TAU, ACTOR_LR, CRITIC_LR, WEIGHT_DECAY, UPDATE_EVERY, N_UPDATES,\
                                    A, INIT_BETA, P_EPS, N_STEPS, SEP_EXP, V_MIN, V_MAX,\
                                    CLIP, N_ATOMS, INIT_SIGMA, LINEAR, FACTORIZED, DISTRIBUTIONAL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agents, seed=SEED, batch_size=BATCH_SIZE,
                 buffer_size=BUFFER_SIZE, start_since=START_SINCE, gamma=GAMMA,
                 tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, weight_decay=WEIGHT_DECAY, update_every=UPDATE_EVERY, n_updates=N_UPDATES,
                 priority_eps=P_EPS, a=A, initial_beta=INIT_BETA, n_multisteps=N_STEPS, separate_experiences=SEP_EXP,
                 v_min=V_MIN, v_max=V_MAX, clip=CLIP, n_atoms=N_ATOMS,
                 initial_sigma=INIT_SIGMA, linear_type=LINEAR, factorized=FACTORIZED, distributional=DISTRIBUTIONAL, **kwds):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            n_agents (int): number of agents, or simulations, running simultaneously
            seed (int): random seed
            batch_size (int): size of each sample batch
            buffer_size (int): size of the experience memory buffer
            start_since (int): number of steps to collect before start training
            gamma (float): discount factor
            tau (float): target network soft-update parameter
            actor_lr (float): learning rate for the actor network
            critic_lr (float): learning rate for the critic network
            weight_decay (float): weight decay for the critic optimizer
            update_every (int): update step—learning and target update—interval
            n_updates (int): number of updates per update step
            priority_eps (float): small base value for priorities
            a (float): priority exponent parameter
            initial_beta (float): initial importance-sampling weight
            n_multisteps (int): number of steps to consider for each experience
            separate_experiences (bool): whether to store experiences with no overlap
            v_min (float): minimum reward support value
            v_max (float): maximum reward support value
            clip (float): gradient norm clipping (`None` to disable)
            n_atoms (int): number of atoms in the discrete support distribution
            initial_sigma (float): initial noise parameter weights
            linear_type (str): one of ('linear', 'noisy'); type of linear layer to use
            factorized (bool): whether to use factorized gaussian noise in noisy layers
            distributional (bool): whether to use distributional learning
        """
        if kwds != {}:
            print("Ignored keyword arguments: ", end='')
            print(*kwds, sep=', ')
        assert isinstance(state_size, int)
        assert isinstance(action_size, int)
        assert isinstance(n_agents, int)
        assert isinstance(seed, int)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(buffer_size, int) and buffer_size >= batch_size
        assert isinstance(start_since, int) and batch_size <= start_since <= buffer_size
        assert isinstance(gamma, (int, float)) and 0 <= gamma <= 1
        assert isinstance(tau, (int, float)) and 0 <= tau <= 1
        assert isinstance(actor_lr, (int, float)) and actor_lr >= 0
        assert isinstance(critic_lr, (int, float)) and critic_lr >= 0
        assert isinstance(weight_decay, (int, float)) and weight_decay >= 0
        assert isinstance(update_every, int) and update_every > 0
        assert isinstance(n_updates, int) and n_updates > 0
        assert isinstance(priority_eps, (int, float)) and priority_eps >= 0
        assert isinstance(a, (int, float)) and 0 <= a <= 1
        assert isinstance(initial_beta, (int, float)) and 0 <= initial_beta <= 1
        assert isinstance(n_multisteps, int) and n_multisteps > 0
        assert isinstance(separate_experiences, bool)
        assert isinstance(v_min, (int, float)) and isinstance(v_max, (int, float)) and v_min < v_max
        if clip: assert isinstance(clip, (int, float)) and clip >= 0
        assert isinstance(n_atoms, int) and n_atoms > 0
        assert isinstance(initial_sigma, (int, float)) and initial_sigma >= 0
        assert isinstance(linear_type, str) and linear_type.strip().lower() in ('linear', 'noisy')
        assert isinstance(factorized, bool)
        assert isinstance(distributional, bool)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.state_size           = state_size
        self.action_size          = action_size
        self.n_agents             = n_agents
        self.seed                 = seed
        self.batch_size           = batch_size
        self.buffer_size          = buffer_size
        self.start_since          = max(start_since, self.batch_size)
        self.gamma                = gamma
        self.tau                  = tau
        self.actor_lr             = actor_lr
        self.critic_lr            = critic_lr
        self.weight_decay         = weight_decay
        self.update_every         = update_every
        self.n_updates            = n_updates
        self.priority_eps         = priority_eps
        self.a                    = a
        self.beta                 = initial_beta
        self.n_multisteps         = n_multisteps
        self.separate_experiences = bool(separate_experiences)
        self.v_min                = v_min
        self.v_max                = v_max
        self.clip                 = clip
        self.n_atoms              = n_atoms
        self.initial_sigma        = initial_sigma
        self.linear_type          = linear_type.strip().lower()
        self.factorized           = bool(factorized)
        self.distributional       = bool(distributional)

        # Distribution
        self.supports = torch.linspace(v_min, v_max, n_atoms, device=device)
        self.delta_z  = (v_max - v_min) / (n_atoms - 1)

        # Actor Networks & Optimizer
        self.actor_local  = Actor(state_size, action_size, linear_type, initial_sigma, factorized).to(device)
        self.actor_target = Actor(state_size, action_size, linear_type, initial_sigma, factorized).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

        # Critic Networks & Optimizer
        self.critic_local  = Critic(state_size, action_size, n_atoms, distributional).to(device)
        self.critic_target = Critic(state_size, action_size, n_atoms, distributional).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr, weight_decay=weight_decay)

        # Replay memory
        self.memory = ReplayBuffer(n_agents, buffer_size, batch_size, n_multisteps, gamma, a, separate_experiences)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.u_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for i_agent, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            self.memory.add(i_agent, state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.u_step = (self.u_step + 1) % self.update_every
        if self.u_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.start_since:
                for _ in range(self.n_updates):
                    experiences, target_discount, is_weights, indices = self.memory.sample(self.beta)
                    new_priorities = self.learn(experiences, is_weights, target_discount)
                    self.memory.update_priorities(indices, new_priorities)

    def reset(self):
        """Prepare the memory for the next episode (Clear n-step collector)"""
        self.memory.reset_multisteps(-1)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        state = torch.tensor(state, dtype=torch.float, device=device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        return action

    def learn(self, experiences, is_weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            is_weights (torch.Tensor): tensor of importance-sampling weights
            gamma (float): discount factor for the target max-Q value

        Returns
        =======
            new_priorities (List[float]): list of new priority values for the given sample
        """
        states, actions, rewards, next_states, dones = experiences

        ### --------------- Critic Update --------------- ###
        ## Distributional ##
        if self.distributional:
            with torch.no_grad():
                next_actions    = self.actor_target(next_states)

                Q_targets_probs = F.softmax(self.critic_target(next_states, next_actions), dim=-1)

                tz_projected    = torch.clamp(rewards + (1 - dones) * gamma * self.supports, min=self.v_min, max=self.v_max)
                b               = tz_projected.sub(self.v_min).div(self.delta_z)
                u               = b.ceil()
                l               = b.floor()
                u_updates       = b.sub(l).add(u.eq(l).type(u.dtype)) # fixes the problem when having b == u == l
                l_updates       = u.sub(b)
                indices_flat    = torch.cat((u.long(), l.long()), dim=1)
                indices_flat    = indices_flat.add(torch.arange(start=0,
                                                                end=b.size(0) * b.size(1),
                                                                step=b.size(1),
                                                                dtype=indices_flat.dtype,
                                                                layout=indices_flat.layout,
                                                                device=indices_flat.device).unsqueeze(1)).view(-1)
                updates_flat = torch.cat((u_updates.mul(Q_targets_probs), l_updates.mul(Q_targets_probs)), dim=1).view(-1)
                Q_target_dists = torch.zeros_like(Q_targets_probs)
                Q_target_dists.view(-1).index_add_(0, indices_flat, updates_flat)

            Q_predicted_dists = self.critic_local(states, actions)

            # critic_loss_per_instance = Q_target_dists.mul(Q_predicted_dists.exp().sum(dim=-1, keepdim=True).log() - Q_predicted_dists).sum(dim=-1, keepdim=False) # Cross-Entropy
            critic_loss_per_instance = Q_target_dists.mul(Q_predicted_dists.exp().sum(dim=-1, keepdim=True).log() - Q_predicted_dists + Q_target_dists.add(Q_target_dists.eq(0).type(Q_target_dists.dtype)).log()).sum(dim=-1, keepdim=False) # KL-Divergence
            # critic_loss_per_instance = F.kl_div(F.log_softmax(Q_predicted_dists, dim=-1), Q_target_dists, reduce=False).sum(dim=-1, keepdim=False) # KL-Divergence
        ## Single Q Value ##
        else:
            with torch.no_grad():
                next_actions = self.actor_target(next_states)
                Q_targets = self.critic_target(next_states, next_actions).mul(dones.neg().add(1)).mul(gamma).add(rewards)

            Q_predicted = self.critic_local(states, actions)

            critic_loss_per_instance = F.mse_loss(Q_predicted, Q_targets, reduce=False).sum(dim=-1, keepdim=False)

        new_priorities = critic_loss_per_instance.detach().add(self.priority_eps).cpu().numpy()
        critic_loss = critic_loss_per_instance.mul(is_weights.view(-1)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip:
            nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.clip)
        self.critic_optimizer.step()

        ### --------------- Actor Update --------------- ###
        pred_actions = self.actor_local(states)
        if self.distributional:
            pred_q_dists = F.softmax(self.critic_local(states, pred_actions), dim=-1)
            pred_q_values = pred_q_dists.mul(self.supports).sum(dim=-1, keepdim=False)
        else:
            pred_q_values = self.critic_local(states, pred_actions)
        actor_loss = pred_q_values.mean().neg()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip:
            nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.clip)
        self.actor_optimizer.step()

        ### --------------- Target Updates --------------- ###
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)

        ### Return updated priorities
        return new_priorities

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def noise(self, enable):
        """Enable or disable noise from all noisy layers."""
        self.actor_local.noise(enable)
        self.actor_target.noise(enable)
