from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict
import random
from itertools import combinations

class Card(Enum):
    Q = 0
    K = 1
    A = 2
    Joker = 3

class LiarsBarEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    TABLE_CARDS = [Card.Q, Card.K, Card.A]
    MAX_CARDS_PER_TURN = 3
    INITIAL_HAND_SIZE = 5
    MIN_DEATH_BULLET = 1
    MAX_DEATH_BULLET = 6
    NUMBER_OF_DISTINCT_RANKS = 4

    def __init__(self, num_players: int = 2):
        super(LiarsBarEnv, self).__init__()

        self.num_players = num_players
        self.players = []
        self.alive_players = []
        self.table_card = None
        self.player_turn = 0
        self.previous_player_index = None
        self.round_finished = False
        self.last_played_cards = None
        self.deck = None

        # Observation space: Dict containing hand, table card, and last played cards
        self.observation_space = spaces.Dict({
            "hand": spaces.MultiDiscrete([self.NUMBER_OF_DISTINCT_RANKS] * self.INITIAL_HAND_SIZE),
            "table_card": spaces.Discrete(len(self.TABLE_CARDS)),
            "last_played": spaces.Discrete(self.MAX_CARDS_PER_TURN + 1),
            "player_turn": spaces.Discrete(num_players)
        })

        # Action space: Discrete actions (0 - challenge, 1..31 all combinations of playing cards)
        self.action_space = spaces.Discrete(2**self.INITIAL_HAND_SIZE)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.players = [self._create_player(i) for i in range(self.num_players)]
        self.alive_players = [player for player in self.players]
        self.reset_round()

        return self._get_observation(), {}
    
    def reset_round(self):
        self.table_card = random.choice(self.TABLE_CARDS)
        self.last_played_cards = []
        self.player_turn = 0
        self.round_finished = False

        self.deck = self._initialize_deck()
        self._deal_cards()

        return self._get_observation()

    def _deal_cards(self):
        for player in self.players:
            player["hand"] = [self.deck.pop() for _ in range(self.INITIAL_HAND_SIZE)]

    def _get_observation(self, current_player: dict = None) -> Dict:
        current_player_hand = current_player["hand"] if current_player else []

        return {
            "hand": current_player_hand,
            "table_card": self.table_card,
            "last_played": len(self.last_played_cards),
            "player_turn": self.player_turn,
            "alive": 0 if current_player not in self.alive_players else 1
        }

    def step(self, action: int):
        current_player = self.alive_players[self.player_turn]

        if action == 0 or self._calculate_active_players_in_round() == 1:  # Challenge
            self._challenge_previous_player()
        else:  # Play cards
            self._play_turn(action)

        reward = self._calculate_reward(current_player)

        if len(self.alive_players) == 1:
            done = True
        elif self.round_finished:
            self.reset_round()
            done = False
        else:
            done = False

        return self._get_observation(current_player), reward, done, {}
    
    def _calculate_active_players_in_round(self):
        return len([player for player in self.alive_players if player["hand"] != [-1, -1, -1, -1, -1]])

    def render(self, mode="human"):
        print(f"Table card: {self.table_card}")
        print(f"Current player turn: {self.player_turn}")
        for i, player in enumerate(self.alive_players):
            print(f"Player {i} {player['bullets_shot']}/{player['death_bullet']}", end=" ")
            for j in range(self.INITIAL_HAND_SIZE):
                print(player["hand"][j], end= " ")
            print()

    def _create_player(self, player_id: int):
        return {
            "id": player_id,
            "hand": [],
            "death_bullet": random.randint(self.MIN_DEATH_BULLET, self.MAX_DEATH_BULLET),
            "bullets_shot": 0
        }

    def _initialize_deck(self):
        cards = [Card.Q] * 6 + [Card.K] * 6 + [Card.A] * 6 + [Card.Joker] * 2
        random.shuffle(cards)
        return cards

    
    def next_player_turn(self):
        """Finds the next active player (player with cards in hand)."""
        next_index = (self.player_turn + 1) % len(self.alive_players)
        while self.alive_players[next_index]["hand"] == [-1, -1, -1, -1, -1]:
            next_index = (next_index + 1) % len(self.alive_players)
        return next_index

    def _get_available_actions(self) -> List[int]:
        """
        Determine all valid actions for the current player given the state of the game.
        Returns:
            A list of tuples, where each tuple represents an action:
            (action_type, cards_to_play)
            - action_type: 0 for play, 1 for challenge
            - cards_to_play: A binary list indicating which cards to play (length matches hand size)
        """
        current_player = self.alive_players[self.player_turn]
        hand = current_player["hand"]

        available_actions = []

        if self.last_played_cards:
            # Challenge
            available_actions.append(0)  # No cards are played in a challenge action

        if self._calculate_active_players_in_round() > 1:
            # Play cards
            for action in range(1, 2**self.INITIAL_HAND_SIZE):
                cards_binary = [int(x) for x in format(action, f"0{self.INITIAL_HAND_SIZE}b")]
                if sum(cards_binary) > self.MAX_CARDS_PER_TURN:
                    continue
                if all(hand[idx] != -1 for idx in range(len(hand)) if cards_binary[idx] == 1):
                    available_actions.append(action)

        return available_actions

    def _play_turn(self, action: int):
        current_player = self.alive_players[self.player_turn]
        cards_binary = [int(x) for x in format(action, f"0{self.INITIAL_HAND_SIZE}b")]
        
        if sum(cards_binary) > self.MAX_CARDS_PER_TURN or sum(cards_binary) == 0:
            raise ValueError("Too many cards played")

        self.last_played_cards = [current_player["hand"][i] for i in range(len(cards_binary)) if cards_binary[i] == 1]
        
        for i, play_card in enumerate(cards_binary):
            if play_card == 1:
                if current_player['hand'][i] == -1:
                    raise ValueError("Card already played")
                current_player["hand"][i] = -1


        self.previous_player_index = self.player_turn
        self.player_turn = self.next_player_turn()

    def _challenge_previous_player(self):
        if self.previous_player_index is None:
            raise ValueError("No previous player to challenge")

        if all(card == self.table_card or card == Card.Joker for card in self.last_played_cards):
            self._apply_bullet(self.alive_players[self.player_turn])
        else:
            self._apply_bullet(self.alive_players[self.previous_player_index])

        self.round_finished = True

    def _apply_bullet(self, player):
        player["bullets_shot"] += 1
        if player["bullets_shot"] == player["death_bullet"]:
            self.alive_players.remove(player)

    def _calculate_reward(self, current_player: dict):
        if current_player not in self.alive_players:
            return -10
        elif len(self.alive_players) == 1:
            return 10
        else:
            return 0
