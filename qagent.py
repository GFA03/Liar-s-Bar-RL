import numpy as np
import random
from collections import defaultdict
import gymnasium as gym

class QLearningAgent:
    def __init__(self, env: gym.Env, learning_rate: float = 0.1, discount_factor: float = 0.9, exploration_rate: float = 1.0, exploration_decay: float = 0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

        # Initialize Q-table
        self.q_table = defaultdict(lambda: {})

    def get_action(self, state):
        available_actions = self.env._get_available_actions()
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        else:
            state_key = self._state_to_key(state)
        return max(available_actions, key=lambda action: self.q_table[state_key].get(action, 0.0))


    def learn(self, state, action, reward, next_state, done):
        """Update Q-table based on the action taken and reward received."""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        print(action)

        # Initialize Q-values for the state-action pair if not already done
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0

        # Compute TD target and update rule
        best_next_action = max(self.q_table[next_state_key], key=self.q_table[next_state_key].get, default=None)
        td_target = reward + (self.discount_factor * self.q_table[next_state_key].get(best_next_action, 0.0) * (not done))
        self.q_table[state_key][action] += self.learning_rate * (td_target - self.q_table[state_key][action])

        if done:
            self.exploration_rate *= self.exploration_decay

    def _state_to_key(self, state):
        """Convert state dictionary to a tuple key for Q-table."""
        hand_key = tuple(state["hand"])
        table_card_key = state["table_card"]
        last_played_key = state["last_played"]
        player_turn_key = state["player_turn"]
        return (hand_key, table_card_key, last_played_key, player_turn_key)
