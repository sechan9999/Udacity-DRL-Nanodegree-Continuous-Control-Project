import math
import time
import numpy as np
import torch
from collections import deque, namedtuple

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, n_agents, buffer_size, batch_size, n_multisteps, gamma, a, separate_experiences):
        """Initialize a ReplayBuffer object.

        Params
        ======
            n_agents (int): number of agents, or simulations, running simultaneously
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            n_multisteps (int): number of time steps to consider for each experience
            gamma (float): discount factor
            a (float): priority exponent parameter
            separate_experiences (bool): whether to store experiences with no overlap
        """
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_multisteps = n_multisteps
        self.gamma = gamma
        self.a = a
        self.separate_experiences = bool(separate_experiences)

        self.sample_times = []
        self.tensorize_times = []
        self.update_times = []

        self._at_least_once_updated = False

        self.memory_write_idx = 0
        self._non_leaf_depth = math.ceil(math.log2(buffer_size))
        self._memory_start_idx = 2 ** self._non_leaf_depth
        self._buffer_is_full = False
        self.memory = [None for _ in range(buffer_size)]
        self.priorities_a = np.zeros(buffer_size)
        self.tree = np.zeros(self._memory_start_idx + buffer_size) # starts from index 1, not 0; makes implementation easier and reduces many small computations

        # self.memory = deque(maxlen=buffer_size)
        # self.priorities_a = deque(maxlen=buffer_size)
        self.multistep_collectors = [deque(maxlen=n_multisteps) for _ in range(n_agents)]
        self.max_priority_a = 1.
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        self._discounts = np.power(self.gamma, np.arange(self.n_multisteps + 1))
        self._target_discount = float(self._discounts[-1])

    def add(self, i_agent, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        assert isinstance(i_agent, int)
        assert 0 <= i_agent < self.n_agents
        collector = self.multistep_collectors[i_agent]
        e = self.experience(state, action, reward, next_state, done)
        collector.append(e)
        if len(collector) == self.n_multisteps:
            self.memory[self.memory_write_idx] = tuple(collector)

            delta_priority_a = self.max_priority_a - self.priorities_a[self.memory_write_idx]
            tree_idx = self._memory_start_idx + self.memory_write_idx
            self.priorities_a[self.memory_write_idx] = self.max_priority_a
            self.tree[tree_idx] = self.max_priority_a
            for _ in range(self._non_leaf_depth):
                tree_idx = tree_idx // 2
                self.tree[tree_idx] += delta_priority_a
            if self.tree[1] < 0:
                raise ValueError(self.tree[:10])

            self.memory_write_idx += 1
            if self.memory_write_idx >= self.buffer_size:
                self._buffer_is_full = True
                self.memory_write_idx = 0

            # self.memory.append(tuple(collector))
            # self.priorities_a.append(self.max_priority_a)
            if self.separate_experiences:
                collector.clear()
        if done:
            collector.clear()

    def sample(self, beta):
        """Randomly sample a batch of experiences from memory.

        Params
        ======
            beta (int or float): parameter used for calculating importance-priority weights

        Returns
        =======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            target_discount (float): discount factor for target max-Q value
            is_weights (torch.Tensor): tensor of importance-sampling weights
            indices (np.ndarray): sample indices"""
        t0 = time.time()

        sample_values = debug = np.linspace(0, self.tree[1], num=self.batch_size, endpoint=False, dtype=np.float32)
        sample_values += np.multiply(np.random.rand(self.batch_size), np.subtract(sample_values[1], sample_values[0]))
        tree_indices = np.ones(self.batch_size, dtype=np.int32)
        for d in range(self._non_leaf_depth):
            left_child_indices = np.multiply(tree_indices, 2)
            right_child_indices = np.add(left_child_indices, 1)
            greater_than_left = np.greater(sample_values, self.tree[left_child_indices])
            sample_values = np.where(greater_than_left, np.subtract(sample_values, self.tree[left_child_indices]), sample_values)
            tree_indices = np.where(greater_than_left, right_child_indices, left_child_indices)
            if d == 19 and False:
                raise Exception(sample_values, tree_indices)
        indices = np.subtract(tree_indices, self._memory_start_idx)

        # probs = np.divide(self.priorities_a, sum(self.priorities_a))

        # indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=probs)

        try:
            experiences = tuple(zip(*[self.memory[i] for i in indices if self.memory[i] is not None]))
        except:
            raise Exception(debug)
        if self.memory_write_idx > 1000:
            raise Exception(indices)
        self.sample_times.append(time.time() - t0)
        t0 = time.time()

        first_states = torch.tensor([e[0] for e in experiences[0]], dtype=torch.float, device=device)
        actions      = torch.tensor([e[1] for e in experiences[0]], dtype=torch.float, device=device)
        rewards      = torch.tensor(
                           np.sum(
                               np.multiply(
                                   np.array([[e[2] for e in experiences_step] for experiences_step in experiences]).transpose(), self._discounts[:-1]
                               ), axis=1, keepdims=True
                           ), dtype=torch.float, device=device)
        last_states  = torch.tensor([e[3] for e in experiences[-1]], dtype=torch.float, device=device)
        dones        = torch.tensor([e[4] for e in experiences[-1]], dtype=torch.float, device=device).view(-1, 1)

        is_weights = np.divide(self.priorities_a[indices], self.tree[1])
        is_weights = np.power(np.multiply(is_weights, self.buffer_size if self._buffer_is_full else self.memory_write_idx), -beta)
        # is_weights = [probs[i] for i in indices if self.memory[i] is not None]
        # is_weights = np.power(np.multiply(is_weights, len(self.memory)), -beta)
        is_weights = torch.tensor(np.divide(is_weights, np.max(is_weights)).reshape((-1, 1)), dtype=torch.float, device=device)
        self.tensorize_times.append(time.time() - t0)

        return (first_states, actions, rewards, last_states, dones), self._target_discount, is_weights, indices

    def update_priorities(self, indices, new_priorities):
        """Update the priorities for the experiences of given indices to the given new values.

        Params
        ======
            indices (array_like): indices of experience priorities to update
            new_priorities (array-like): new priority values for given indices"""
        t0 = time.time()
        new_priorities_a = np.power(new_priorities, self.a)

        if not self._at_least_once_updated:
            self._at_least_once_updated = True
            max_a = np.max(new_priorities_a)

            if self._buffer_is_full:
                self.priorities_a[:] = max_a
                idx_start = self._memory_start_idx
                idx_end = len(self.tree)

            else:
                self.priorities_a[:self.memory_write_idx] = max_a
                idx_start = self._memory_start_idx
                idx_end = self._memory_start_idx+self.memory_write_idx

            self.tree[:idx_start] = 0.
            self.tree[idx_start:idx_end] = max_a

            for _ in range(self._non_leaf_depth - 1):
                child_indices = np.arange(idx_start, idx_end)
                parent_indices = np.floor_divide(child_indices, 2)
                np.add.at(self.tree, parent_indices, self.tree[child_indices])
                idx_start //= 2
                idx_end = ((idx_end - 1) // 2) + 1

        debugs = {'indices': np.copy(indices), 'new_priorities': np.copy(new_priorities),
                  'tree': np.copy(self.tree), 'priorities_a': np.copy(self.priorities_a),
                  'memory_write_idx': self.memory_write_idx, 'memory_start_idx': self._memory_start_idx}
        torch.save(debugs, 'debugs.var')
        raise Exception()

        delta_priority_a = np.subtract(new_priorities_a, self.priorities_a[indices])
        parent_indices = np.add(indices, self._memory_start_idx)
        self.priorities_a[indices] = new_priorities_a
        self.tree[parent_indices] = new_priorities_a
        for _ in range(self._non_leaf_depth - 1):
            child_indices = parent_indices
            parent_indices = np.floor_divide(child_indices, 2)
            np.add.at(self.tree, parent_indices, delta_priority_a) # propagate changes in values to parent nodes
            if np.any(self.tree < 0):
                np.add.at(self.tree, parent_indices, -delta_priority_a)
                raise RuntimeError(self.tree[parent_indices], parent_indices, child_indices, delta_priority_a)
        self.tree[[0, 1]] = self.tree[2] + self.tree[3]
        # for i, new_priority_a in zip(indices, new_priorities_a):
        #     self.priorities_a[i] = float(new_priority_a)
        self.max_priority_a = np.max(self.priorities_a)
        self.update_times.append(time.time() - t0)

    def reset_multisteps(self, i_agent=-1):
        assert isinstance(i_agent, int) and -1 <= i_agent < self.n_agents
        if i_agent == -1:
            for collector in self.multistep_collectors:
                collector.clear()
        else:
            self.multistep_collectors[i_agent].clear()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer_size) if self._buffer_is_full else self.memory_write_idx
