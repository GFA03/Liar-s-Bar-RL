from typing import List
from card import Card
from deck import Deck
import random

from player import Player

class LiarsBarEnv:
	TABLE_CARDS = ['Q', 'K', 'A']

	def __init__(self) -> None:
		self.players = []
		self.table_card = None
		self.player_turn = 0
		self.round_finished = False

	def add_player(self, player) -> None:
		self.players.append(player)

	def next_player_turn(self) -> int:
		return (self.player_turn + 1) % len(self.players)
	
	def previous_player_turn(self) -> int:
		return (self.player_turn - 1) % len(self.players)

	def get_current_player(self) -> Player:
		return self.players[self.player_turn]
	
	def remove_player(self, player: Player) -> None:
		player_index = self.players.index(player)
		self.players.remove(player)
		self.bullets.pop(player_index)
		self.death_bullet.pop(player_index)

	def deal_cards(self) -> None:
		for player in self.players:
			player.hand = [self.deck.draw() for _ in range(5)]

	def initialize_round(self) -> None:
		self.deck = Deck()
		self.deck.shuffle()
		self.bullets = [0] * len(self.players)
		self.death_bullet = [random.randint(1, 6) for i in range(len(self.players))]
		self.deal_cards()
		self.table_card = random.choice(self.TABLE_CARDS)
		self.last_played_cards = None
		self.round_finished = False

	def is_last_player_with_a_hand(self) -> bool:
		count_hands = sum([1 for p in self.players if len(p.hand) > 0])
		return count_hands <= 1

	def play_turn(self, cards: List[Card]) -> bool:
		if cards is None or len(cards) == 0 or self.is_last_player_with_a_hand():
			print("Invalid turn played")
			return False
		self.last_played_cards = cards
		for card in cards:
			self.get_current_player().hand.remove(card)

		self.player_turn = self.next_player_turn()
		return True
		

	def check_last_played_cards(self) -> bool:
		for card in self.last_played_cards:
			if card.rank != self.table_card and card.rank != 'Joker':
				return False
		return True

	def challenge_last_player(self) -> bool:
		if self.last_played_cards is None:
			print("Invalid challenge played")
			return False
		self.round_finished = True
		if self.check_last_played_cards() is True:
			print(f"Player {self.get_current_player().name} has lost! You will be shot now!")
			self.shoot_player(self.get_current_player())
		else:
			print(f"Player {self.players[self.previous_player_turn()].name} has lost! You will be shot now!")
			self.shoot_player(self.players[self.previous_player_turn()])
		return True
	
	def shoot_player(self, player: Player) -> None:
		player_index = self.players.index(player)
		self.bullets[player_index] += 1
		if (self.death_bullet[player_index] == self.bullets[player_index]):
			self.remove_player(player)
			print(f"Player {player.name} has been shot")
			self.player_turn = 0
		else:
			self.player_turn = player_index
	