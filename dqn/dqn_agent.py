import random
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Dict, List

from monte_carlo.mc_env import LiarsBarEdiEnv


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 3)
        self.out = nn.Linear(hidden_size // 3, 1)

    def forward(self, state_action):
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        q_value = self.out(x)
        return q_value

    def save(self, path):
        torch.save(self.model.state_dict(), path)

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def store(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
            self,
            env: LiarsBarEdiEnv,
            gamma=0.9,
            epsilon=0.1,
            lr=1e-3,
            batch_size=32,
            buffer_capacity=20000
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size

        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.state_dim = 25
        self.action_dim = 4

        # model and optimizer
        self.q_network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_size=64
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def encode_state(self, state: Dict) -> np.ndarray:
        """
        Encode the state in a np.array of shape (25,)
        - hand       -> 4 ints
        - table_card -> 1 int
        - history    -> 20 ints ,concatenate 0 if it's shorter
        """
        hand = state["hand"]
        table_card = [state["table_card"]]
        history = list(state["history"])
        history = history + [0] * (20 - len(history))

        state_vec = np.array(hand + table_card + history, dtype=np.float32)
        return state_vec

    def encode_action(self, action: List[int]) -> np.ndarray:
        """
        Encode the action in a np.array of shape (4,)
        """
        return np.array(action, dtype=np.float32)

    def choose_action(self, state: Dict) -> List[int]:
        """
        Epsilon-greedy: with probability epsilon, choose a random valid action,
                        else choose maximum Q action.
        """
        available_actions = self.env._get_available_actions()

        if np.random.rand() < self.epsilon:
            return random.choice(available_actions)
        else:
            state_vec = self.encode_state(state)
            best_action = None
            best_q = -1e9

            for a in available_actions:
                a_vec = self.encode_action(a)
                sa = np.concatenate([state_vec, a_vec], axis=0)
                sa_t = torch.tensor(sa, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_val = self.q_network(sa_t).item()
                if q_val > best_q:
                    best_q = q_val
                    best_action = a
            return best_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def train_step(self):
        """
        One step of update of the weights over a mini batch from buffer
        """
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for (s, a, r, s_next, d) in batch:
            state_batch.append(self.encode_state(s))
            action_batch.append(self.encode_action(a))
            reward_batch.append(r)
            next_state_batch.append(self.encode_state(s_next))
            done_batch.append(d)

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).unsqueeze(-1)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32)
        done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32).unsqueeze(-1)

        # concatenate into (s, a) pairs
        sa_input = torch.cat([state_batch, action_batch], dim=1)
        q_values = self.q_network(sa_input)

        # target: r + gamma * max_{a'}Q(s', a')
        target_vals = []
        for i in range(self.batch_size):
            r_i = reward_batch[i].item()
            done_i = done_batch[i].item()

            if done_i == 1.0:
                # if it s a terminal state, q_value is the reward
                y = r_i
            else:
                s_next_vec = next_state_batch[i].numpy()
                best_q_ns = -1e9

                # generate all states
                candidate_actions = []
                for jokers in range(4):
                    for valets in range(4):
                        for queens in range(4):
                            for kings in range(4):
                                if jokers + valets + queens + kings <= 3:
                                    candidate_actions.append([jokers, valets, queens, kings])

                # compute q_value for each state and extract maximum
                for ca in candidate_actions:
                    ca_vec = self.encode_action(ca)
                    sa_next = np.concatenate([s_next_vec, ca_vec], axis=0)
                    sa_next_t = torch.tensor(sa_next, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        q_val_ns = self.q_network(sa_next_t).item()
                    if q_val_ns > best_q_ns:
                        best_q_ns = q_val_ns

                y = r_i + self.gamma * best_q_ns

            target_vals.append([y])

        target_vals = torch.tensor(target_vals, dtype=torch.float32)

        loss = self.loss_fn(q_values, target_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state: Dict) -> List[int]:
        """
        Choose best action
        """
        available_actions = self.env._get_available_actions()
        state_vec = self.encode_state(state)
        best_action = None
        best_q = -1e9
        for a in available_actions:
            a_vec = self.encode_action(a)
            sa = np.concatenate([state_vec, a_vec], axis=0)
            sa_t = torch.tensor(sa, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_val = self.q_network(sa_t).item()
            if q_val > best_q:
                best_q = q_val
                best_action = a
        return best_action

