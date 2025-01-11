import random
from typing import Dict
from xmlrpc.client import MININT

import numpy as np

from monte_carlo.mc_env import LiarsBarEdiEnv


class MonteCarloAgent:
    def __init__(self, env: LiarsBarEdiEnv, epsilon: float = 0.1, gamma: float = 0.9):
        self.env = env
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma      # Discount factor
        self.Q = {}  # State-action value table
        self.returns = {}  # State-action returns (to calculate average reward)

    def _get_state_key(self, state):
        """Generates a key for state-action pair."""
        return tuple(state["hand"]), state["table_card"], tuple(state["history"])

    def choose_action(self, state):
        """Select action using epsilon-greedy strategy."""
        available_actions = self.env._get_available_actions()
        state_key = tuple(state["hand"]), state["table_card"], tuple(state["history"])

        # If no actions have been taken from this state, explore randomly
        if state_key not in self.Q:
            self.Q[state_key] = {tuple(a): 0.0 for a in available_actions}
            self.returns[state_key] = {tuple(a): [] for a in available_actions}

        if random.random() < self.epsilon:
            # Exploration: randomly choose an action
            action = random.choice(available_actions)
        else:
            # Exploitation: choose the action with the highest Q value
            action = max(available_actions, key=lambda a: self.Q[state_key].get(tuple(a), 0.0))

        return action

    def learn(self, episode):
        G = 0
        for step_state in reversed(episode):
            reward = step_state["reward"]
            state = step_state["state"]
            action = step_state["action"]
            G = reward + self.gamma * G  # Discounted reward
            state_key = self._get_state_key(state)
            action_key = tuple(action)

            # Update the state-action value
            if state_key not in self.returns:
                self.returns[state_key] = {}
            if state_key not in self.Q:
                self.Q[state_key] = {}
            if action_key not in self.returns[state_key]:
                self.returns[state_key][action_key] = []
            if action_key not in self.Q[state_key]:
                self.Q[state_key][action_key] = 0
            self.returns[state_key][action_key].append(G)

            # Average out all rewards for the state-action pair
            self.Q[state_key][action_key] = np.mean(self.returns[state_key][action_key])

    def act(self, state):
        """Choose the best action for the given state using the learned policy."""
        state_key = self._get_state_key(state)

        available_actions = self.env._get_available_actions()

        if state_key not in self.Q:
            return random.choice(available_actions)

        # Exploit learned policy (choose action with highest Q value)
        return max(available_actions, key=lambda a: self.Q[state_key].get(tuple(a), MININT))
