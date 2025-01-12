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
        - epsilon: rata de explorare (epsilon-greedy)
        - gamma: factor de discount
        - alpha: rata de învățare (learning rate)
        """
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        # Q va fi un dict de forma: Q[state_key][action_key] = valoarea Q
        self.Q: Dict = {}

    def _get_state_key(self, state):
        """Generates a key for the state."""
        return (
            tuple(state["hand"]),
            state["table_card"],
            tuple(state["history"])
        )

    def _init_state_if_needed(self, state_key):
        """Dacă starea nu e în Q, inițializează cu 0 pentru fiecare acțiune posibilă."""
        if state_key not in self.Q:
            available_actions = self.env._get_available_actions()
            self.Q[state_key] = {tuple(a): 0.0 for a in available_actions}

    def choose_action(self, state):
        """
        Selectează o acțiune folosind strategia epsilon-greedy
        față de funcția Q curentă.
        """
        state_key = self._get_state_key(state)
        self._init_state_if_needed(state_key)

        if random.random() < self.epsilon:
            # Explorare: alege aleator acțiunea
            action = random.choice(list(self.Q[state_key].keys()))
        else:
            # Exploatare: alege acțiunea cu valoarea Q maximă
            action = max(
                self.Q[state_key],
                key=lambda a: self.Q[state_key][a]
            )
        return list(action)  # convertim tuple->list (sau cum e necesar în env)

    def learn(self, episode):
        """
        Actualizăm Q pentru fiecare pas din episod prin formula SARSA:

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

        unde a' e acțiunea real aleasă în starea s' (nu cea optimă).

        Parametrul 'episode' poate fi o listă de dict-uri de forma:
           {
             "state": state_dict,
             "action": list_sau_tuple_de_actiune,
             "reward": float
           }
        Ca să putem face SARSA, avem nevoie și de acțiunea următoare (a').
        """
        # Pentru a putea face update în stil SARSA, vom parcurge
        # episodul cu index i și i+1 (fiind starea + starea următoare)
        if len(episode) == 0:
            return

        for i in range(len(episode) - 1):
            current_step = episode[i]
            next_step = episode[i + 1]

            s = current_step["state"]
            a = tuple(current_step["action"])  # cheie în Q
            r = current_step["reward"]

            s_next = next_step["state"]
            a_next = tuple(next_step["action"])  # acțiunea efectiv aleasă

            # Construim cheile pentru Q
            s_key = self._get_state_key(s)
            s_next_key = self._get_state_key(s_next)

            # Inițializăm stările dacă nu există
            self._init_state_if_needed(s_key)
            self._init_state_if_needed(s_next_key)

            # SARSA update
            td_target = r + self.gamma * self.Q[s_next_key][a_next]
            td_error = td_target - self.Q[s_key][a]
            self.Q[s_key][a] += self.alpha * td_error

        # Ultimul pas din episod (dacă e terminal, nu mai are s'+a')
        # Sau dacă vrei să contezi și ultima recompensă direct
        last_step = episode[-1]
        s = last_step["state"]
        a = tuple(last_step["action"])
        r = last_step["reward"]
        s_key = self._get_state_key(s)

        # Pentru stările terminale, Q(s, a) <- Q(s, a) + alpha * [r - Q(s, a)]
        # dacă starea chiar este terminală. (Dacă nu e terminală,
        # poate e tăiat episodul altfel - atenție la cum definești 'episode'.)
        self._init_state_if_needed(s_key)
        # Considerăm terminal: Q(s, a) = Q(s, a) + α * [r - Q(s, a)]
        self.Q[s_key][a] += self.alpha * (r - self.Q[s_key][a])

    def act(self, state):
        """
        Returnează acțiunea (sub forma list) cu cea mai mare valoare Q
        (politica greedy față de Q).
        """
        state_key = self._get_state_key(state)
        self._init_state_if_needed(state_key)

        best_action = max(
            self.Q[state_key],
            key=lambda a: self.Q[state_key][a]
        )
        return list(best_action)
