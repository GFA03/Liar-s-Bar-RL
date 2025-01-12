import random
from typing import Dict

from monte_carlo.mc_env import LiarsBarEdiEnv


class SarsaAgent:
    def __init__(
            self,
            env: LiarsBarEdiEnv,
            epsilon: float = 0.1,
            gamma: float = 0.9,
            alpha: float = 0.1
    ):
        """
        SARSA Agent:
        - epsilon: rata de explorare
        - gamma: factor de discount
        - alpha: rata de învățare
        """
        self.name = "SARSAAgent"
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.Q: Dict = {}

    def _get_state_key(self, state):
        """Generates a key for the state."""
        return (
            tuple(state["hand"]),
            state["table_card"],
            tuple(state["history"])
        )

    def _init_state_if_needed(self, state_key, state):
        if state_key not in self.Q:
            available_actions = LiarsBarEdiEnv.get_available_actions(state)
            self.Q[state_key] = {tuple(a): 0.0 for a in available_actions}

    def choose_action(self, state):
        state_key = self._get_state_key(state)
        self._init_state_if_needed(state_key, state)

        if random.random() < self.epsilon:
            action = random.choice(list(self.Q[state_key].keys()))
        else:
            action = max(
                self.Q[state_key],
                key=lambda a: self.Q[state_key][a]
            )
        return list(action)

    def learn(self, episode):
        """
        Actualizăm Q pentru fiecare pas din episod prin formula SARSA:

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

        unde a' e acțiunea real aleasă în starea s' (nu cea optimă).
        """

        if len(episode) == 0:
            return

        for i in range(len(episode) - 1):
            current_step = episode[i]
            next_step = episode[i + 1]

            s = current_step["state"]
            a = tuple(current_step["action"])
            r = current_step["reward"]

            s_next = next_step["state"]
            a_next = tuple(next_step["action"])

            s_key = self._get_state_key(s)
            s_next_key = self._get_state_key(s_next)

            self._init_state_if_needed(s_key, s)
            self._init_state_if_needed(s_next_key, s)

            # SARSA update
            td_target = r + self.gamma * self.Q[s_next_key][a_next]
            td_error = td_target - self.Q[s_key][a]
            self.Q[s_key][a] += self.alpha * td_error

        # Ultimul pas din episod (dacă e terminal, nu mai are s'+a')
        last_step = episode[-1]
        s = last_step["state"]
        a = tuple(last_step["action"])
        r = last_step["reward"]
        s_key = self._get_state_key(s)

        self._init_state_if_needed(s_key, s)
        self.Q[s_key][a] += self.alpha * (r - self.Q[s_key][a])

    def act(self, state):
        state_key = self._get_state_key(state)
        self._init_state_if_needed(state_key, state)

        best_action = max(
            self.Q[state_key],
            key=lambda a: self.Q[state_key][a]
        )
        return list(best_action)


class SarsaTrainer:
    def __init__(self, env: LiarsBarEdiEnv, agent: SarsaAgent):
        self.env = env
        self.agent = agent

    def train(self, episodes = 100):
        for episode_number in range(episodes):
            self.env.reset()

            done = False
            while not done:
                state = self.env.get_obs()
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)


            # After episode ends, update Q values based on rewards
            episode = self.env.get_player_reward_history()
            for i in range(4):
                self.agent.learn(episode[i])

            print(f"Finished episode {episode_number}")
