from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict
import random
from itertools import combinations

class Card(Enum):
    Joker = 0
    Q = 1
    K = 2
    A = 3

class QLearningEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    TABLE_CARDS = [Card.Q, Card.K, Card.A]
    MAX_CARDS_PER_TURN = 3
    INITIAL_HAND_SIZE = 5
    MIN_DEATH_BULLET = 1
    MAX_DEATH_BULLET = 6
    NUMBER_OF_DISTINCT_RANKS = 4
    EMPTY_HAND = [0, 0, 0, 0]

    def __init__(self, num_players: int = 2):
        super(QLearningEnv, self).__init__()

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
            "hand": spaces.MultiDiscrete([self.INITIAL_HAND_SIZE + 1] * self.NUMBER_OF_DISTINCT_RANKS),
            "table_card": spaces.Discrete(len(self.TABLE_CARDS) + 1),
            "last_played": spaces.Discrete(self.MAX_CARDS_PER_TURN + 1)
        })

        # Action space: Discrete actions (0 - challenge, 1..31 all combinations of playing cards)
        self.action_space = spaces.MultiDiscrete([self.MAX_CARDS_PER_TURN + 1] * self.NUMBER_OF_DISTINCT_RANKS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.players = [self._create_player(i) for i in range(self.num_players)]
        self.alive_players = [True for _ in self.players]
        self.reset_round()

        return self._get_observation(), {}
    
    def reset_round(self):
        self.table_card = random.choice(self.TABLE_CARDS)
        self.last_played_cards = []
        self.player_turn = self.previous_player_index if self.previous_player_index is not None else 0
        if self.alive_players[self.player_turn] == False:
            self.player_turn = self.next_player_turn()
        self.previous_player_index = None
        self.round_finished = False

        self.deck = self._initialize_deck()
        self._deal_cards()

        return self._get_observation()

    def _deal_cards(self):
        for i, player in enumerate(self.players):
            if self.alive_players[i]:
                self._players[i]["hand"] = [0 for _ in range(4)]
                for _ in range(5):
                    self._players[i]["hand"][self.deck.pop()] += 1

    def _get_observation(self) -> Dict:
        current_player_hand = self.players[self.player_turn]["hand"]

        return {
            "hand": current_player_hand,
            "table_card": self.table_card,
            "last_played": sum(self.last_played_cards)
        }

    def step(self, action: int):
        current_player = self.players[self.player_turn]

        if action == [0, 0, 0, 0]:  # Challenge
            self._challenge_previous_player()
        else:  # Play cards
            self._play_turn(action)

        reward = self._calculate_reward(current_player)
        observation = self._get_observation()

        # Check if there is a winner
        if sum(self.alive_players) == 1:
            done = True
        elif self.round_finished:
            self.reset_round()
            done = False
        else:
            self.previous_player_index = self.player_turn
            self.player_turn = self.next_player_turn()
            done = False

        return observation, reward, done, {}
    

    def _play_turn(self, action: List[int]):
        current_player = self.players[self.player_turn]
        
        if sum(action) > self.MAX_CARDS_PER_TURN or sum(action) == 0:
            raise ValueError("Too many cards played")
        if any(current_player["hand"][card] < action[card] for card in range(4)):
            raise ValueError("Player doesn't have all those cards")

        self.last_played_cards = action.copy()
        
        for i in range(4):
            current_player["hand"][i] -= action[i]

    def _challenge_previous_player(self):
        if self.previous_player_index is None:
            raise ValueError("No previous player to challenge")

        if self.table_card == 1 and (self.last_played_cards[2] > 0 or self.last_played_cards[3] > 0) or self.table_card == 2 and (self.last_played_cards[1] > 0 or self.last_played_cards[3] > 0) or self.table_card == 3 and (self.last_played_cards[1] > 0 or self.last_played_cards[2] > 0):
            self._apply_bullet(self.players[self.previous_player_index])
        else:
            self._apply_bullet(self.players[self.player_turn])

        self.round_finished = True

    
    def _calculate_active_players_in_round(self):
        return len([player for player in self.players if player["hand"] != self.EMPTY_HAND and self.alive_players[player["id"]]])

    def render(self, mode="human"):
        print(f"Table card: {self.table_card}")
        print(f"Current player turn: {self.player_turn}")
        for i, player in enumerate(self.players):
            if not self.alive_players[i]:
                continue
            print(f"Player {player["id"]} {player['bullets_shot']}/{player['death_bullet']}", end=" ")
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
        next_index = (self.player_turn + 1) % len(self.players)
        while self.players[next_index]["hand"] == self.EMPTY_HAND or not self.alive_players[next_index]:
            next_index = (next_index + 1) % len(self.players)
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
        current_player = self.players[self.player_turn]
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

    def _apply_bullet(self, player):
        player["bullets_shot"] += 1
        if player["bullets_shot"] == player["death_bullet"]:
            self.alive_players[player["id"]] = False

    def _calculate_reward(self, current_player: dict):
        """
        Calculates the reward for the current player based on game outcomes and actions.
        Returns:
            float: Reward value for the current player.
        """
        # Base rewards
        if not self.alive_players[current_player["id"]]:  # Player is eliminated
            return -50  # Large penalty for elimination
        
        if sum(self.alive_players) == 1 and self.alive_players[current_player["id"]]:  # Player wins the game
            return 100  # Large reward for winning
        
        reward = 0  # Default reward

        # Intermediate rewards
        if self.round_finished:
            if current_player["hand"] == self.EMPTY_HAND:  # Hand is empty
                reward += 20  # Bonus for finishing hand
            if self.previous_player_index is not None and current_player == self.players[self.previous_player_index]:
                if all(card == self.table_card or card == Card.Joker for card in self.last_played_cards):
                    reward += 15  # Reward for successfully bluffing or playing matching cards
                else:
                    reward -= 10  # Penalty for unsuccessful play/challenge

        # Participation rewards
        if len(self.last_played_cards) > 0:
            reward += 5  # Slight reward for active participation

        return reward

