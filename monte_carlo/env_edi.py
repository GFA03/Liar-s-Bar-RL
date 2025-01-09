import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, List


class LiarsBarEdiEnv(gym.Env):
    WIN_REWARD = 100
    LOSS_REWARD = -100
    CARD_PLACED_REWARD = 10
    CORRECT_CHALLENGE_REWARD = 100

    def __init__(self, num_players: int = 4):
        super(LiarsBarEdiEnv, self).__init__()

        self._num_players = num_players
        self._players = []
        self._previous_player_index = None
        self._current_player_index = None
        self._previous_action = None
        self._table_card = None
        self._history = []
        self._player_reward_history = []
        self._number_of_finished_players = 0

        self._observation_space = spaces.Dict({
            "hand": spaces.MultiDiscrete([6] * 4),
            "table_card": spaces.Discrete(4),
            "history": spaces.MultiDiscrete([4] * 20),
        })

        self.action_space = spaces.MultiDiscrete([4] * 4)

    def reset(self, seed=None, options=None):
        super(LiarsBarEdiEnv, self).reset()

        self._current_player_index = np.random.randint(0, self._num_players)
        self._previous_player_index = None
        self._number_of_finished_players = 0
        self._table_card = np.random.choice([1, 2, 3])
        self._history = []
        deck = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        np.random.shuffle(deck)
        self._players = [{} for _ in range(self._num_players)]
        self._player_reward_history = [[] for _ in range(self._num_players)]
        for i in range(self._num_players):
            self._players[i]["hand"] = [0 for _ in range(4)]
            for _ in range(5):
                self._players[i]["hand"][deck.pop()] += 1

        return self._get_obs(), {}

    def _get_obs(self) -> Dict:
        hand = self._players[self._current_player_index]["hand"]

        return {
            "hand": hand,
            "table_card": self._table_card,
            "history": self._history,
        }


    def get_obs(self):
        return self._get_obs()

    def get_player_reward_history(self):
        return self._player_reward_history

    def step(self, action: Tuple[int, int, int, int]):
        self._player_reward_history[self._current_player_index].append({
            "state": self._get_obs(),
            "action": action,
            "reward": 0,
        })
        obs = self._get_obs()
        #Challenge
        if action == [0, 0, 0, 0]:
            self._challenge()
        else:
            self._play_turn(action)

        reward = self._player_reward_history[self._current_player_index][-1]["reward"]


        done = self._check_round_finished(action)

        self._previous_player_index = self._current_player_index
        self._current_player_index = (self._current_player_index + 1) % self._num_players
        while sum(self._players[self._current_player_index]["hand"]) == 0:
            self._current_player_index = (self._current_player_index + 1) % self._num_players

        self._previous_action = action
        return obs, reward, done, {}

    def _challenge(self):
        if self._previous_player_index is None:
            raise ValueError("No previous player to challenge")

        if self._table_card == 1 and (self._previous_action[2] > 0 or self._previous_action[3] > 0):
           self._punish_player(self._previous_player_index)
           self._reward_correct_challenge(self._current_player_index)
        elif self._table_card == 2 and (self._previous_action[1] > 0 or self._previous_action[3] > 0):
           self._punish_player(self._previous_player_index)
           self._reward_correct_challenge(self._current_player_index)
        elif self._table_card == 3 and (self._previous_action[2] > 0 or self._previous_action[1] > 0):
           self._punish_player(self._previous_player_index)
           self._reward_correct_challenge(self._current_player_index)
        else:
            self._punish_player(self._current_player_index)

    def _punish_player(self, player_index):
        self._player_reward_history[player_index][-1]["reward"] += LiarsBarEdiEnv.LOSS_REWARD

    def _reward_correct_challenge(self, player_index):
        self._player_reward_history[player_index][-1]["reward"] += LiarsBarEdiEnv.CORRECT_CHALLENGE_REWARD

    def _play_turn(self, action):
        current_player = self._players[self._current_player_index]

        if self._number_of_finished_players == self._num_players - 1 and action != [0, 0, 0, 0]:
            raise ValueError("You are the last player, you have to challenge")
        if any(current_player["hand"][card] < action[card] for card in range(4)):
            raise ValueError("Player doesn't have all those cards")
        if sum(action) > 3:
            raise ValueError("Too many cards played")

        self._player_reward_history[self._current_player_index][-1]["reward"] += sum(action) * LiarsBarEdiEnv.CARD_PLACED_REWARD

        self._history.append(sum(action))

        for i in range(4):
            current_player["hand"][i] -= action[i]
        if sum(current_player["hand"]) == 0:
            self._number_of_finished_players += 1

    def _check_round_finished(self, action) -> bool:
        return action == [0, 0, 0, 0]

    def _get_available_actions(self) -> List[List[int]]:
        current_player = self._players[self._current_player_index]
        hand = current_player["hand"]
        available_actions = []

        # Challenge
        if self._previous_player_index is not None:
            available_actions.append([0, 0, 0, 0])

        if self._number_of_finished_players == self._num_players - 1:
            return available_actions
        
        for jokers in range(hand[0] + 1):
            for valets in range(hand[1] + 1):
                for queens in range(hand[2] + 1):
                    for kings in range(hand[3] + 1):
                        if 1 <= jokers + valets + queens + kings <= 3:
                            available_actions.append([jokers, valets, queens, kings])

        return available_actions







