import random
from copy import copy

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, List

from numpy.random import logseries

from monte_carlo.mc_env import LiarsBarEdiEnv


class LiarsBarGame:
    def __init__(self, logs = False):
        self._agents = []
        self._agent_lives = []
        self._starting_player_index = 0
        self._logs = logs

    def register_agent(self, agent):
        self._agents.append(agent)
        return self._agents.index(agent)

    def run_game(self):
        self.initialize_game_state()

        while self._at_least_two_players_alive():
            if self._logs:
                print("*** Starting new round ***")

            current_players = self._get_alive_players()

            game_round = LiarsBarRound(current_players, self._starting_player_index, self._logs)
            loser_index = game_round.run_round()

            self._shoot_player(self._agents.index(current_players[loser_index]))
            if self._logs:
                print(f"Lives: {self._agent_lives}")
                print("*** Ending round ***")

            self._set_next_starting_player()

        return self._get_winner_index()



    def initialize_game_state(self):
        self._agent_lives = [random.choice([1, 2, 3, 4, 5, 6]) for _ in self._agents]

    def _get_alive_players(self):
        players = []

        for index, agent in enumerate(self._agents):
            if self._agent_lives[index] > 0:
                players.append(agent)

        return players

    def _at_least_two_players_alive(self):
        alive = 0
        for lives in self._agent_lives:
            if lives > 0:
                alive += 1
        return alive >= 2

    def _shoot_player(self, index):
        self._agent_lives[index] -= 1

    def _get_winner_index(self):
        for index, agent in enumerate(self._agents):
            if self._agent_lives[index] > 0:
                return index

    def _set_next_starting_player(self):
        while True:
            self._starting_player_index += 1
            self._starting_player_index %= len(self._agents)
            if self._agent_lives[self._starting_player_index] > 0:
                cnt = 0
                for i in range(self._starting_player_index):
                    if self._agent_lives[i] <= 0:
                        cnt += 1
                self._starting_player_index -= cnt
                break


class LiarsBarRound:
    def __init__(self, players, starting_player, logs = False):
        self._players = players
        self._generate_hands()
        self._current_player = starting_player
        self._previous_player = None
        self._previous_action = None
        self._table_card = random.choice([0, 1, 2, 3])
        self._history = []
        self._loser = None
        self._logs = logs

    def run_round(self):
        while True:
            state = self._get_state()
            action = self._players[self._current_player].act(state)


            if self._logs:
                print(f"Player: {self._current_player} - {self._players[self._current_player].name} has hand: {self._hands[self._current_player]} and played: {action}")

            self._perform_action(action)

            if self._loser is not None:
                return self._loser



    def _generate_hands(self):
        deck = [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
        self._hands = []

        for _ in self._players:
            hand = [0, 0, 0, 0]
            for _ in range(5):
                card = random.choice(deck)
                deck.pop(deck.index(card))
                hand[card] += 1
            self._hands.append(hand)

    def _get_state(self) -> Dict:
        hand = self._hands[self._current_player]

        return {
            "num_players": len(self._players),
            "hand": copy(hand),
            "table_card": copy(self._table_card),
            "history": copy(self._history),
        }

    def _perform_action(self, action):
        if action == [0, 0, 0, 0]:
            self._challenge()
        else:
            self._play_cards(action)

        self._previous_player = self._current_player
        self._previous_action = action

        self._current_player += 1
        self._current_player %= len(self._players)
        while sum(self._hands[self._current_player]) == 0:
            self._current_player += 1
            self._current_player %= len(self._players)

    def _challenge(self):
        if sum(self._previous_action) != self._previous_action[self._table_card] + self._previous_action[0]:
            self._loser = self._previous_player
            if self._logs:
                print(f"Player: {self._previous_player} - {self._players[self._previous_player].name} has been caught lying")
        else:
            self._loser = self._current_player
            if self._logs:
                print(f"Player: {self._current_player} - {self._players[self._current_player].name} has challenged incorrectly")

    def _play_cards(self, action):
        for i in range(4):
            self._hands[self._current_player][i] -= action[i]

        self._history.append(sum(action))













