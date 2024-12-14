from typing import List
from card import Card
from deck import Deck
import random

from player import Player

class LiarsBarEnv:
	TABLE_CARDS = ['Q', 'K', 'A']
	MAX_CARDS_PER_TURN = 3
	INITIAL_HAND_SIZE = 5
	MIN_DEATH_BULLET = 1
	MAX_DEATH_BULLET = 6

	def __init__(self) -> None:
		self.players = []
		self.table_card = None
		self.player_turn = 0
		self.round_finished = False
		self.last_played_cards = None
		self.deck = None

	def add_player(self, player) -> None:
		self.players.append(player)

	def next_player_turn(self) -> int:
		return (self.player_turn + 1) % len(self.players)
	
	def previous_player_turn(self) -> int:
		return (self.player_turn - 1) % len(self.players)

	def get_current_player(self) -> Player:
		return self.players[self.player_turn]
	
	def remove_player(self, player: Player) -> None:
		self.players.remove(player)

	def deal_cards(self) -> None:
		for player in self.players:
			player.hand = [self.deck.draw() for _ in range(self.INITIAL_HAND_SIZE)]

	def initialize_round(self) -> None:
		self.deck = Deck()
		self.deck.shuffle()
		self.assign_death_bullets()
		self.deal_cards()
		self.table_card = random.choice(self.TABLE_CARDS)
		self.last_played_cards = None
		self.round_finished = False

	def assign_death_bullets(self) -> None:
		for player in self.players:
			player.death_bullet = random.randint(self.MIN_DEATH_BULLET, self.MAX_DEATH_BULLET)

	def is_last_player_with_a_hand(self) -> bool:
		active_players = [player for player in self.players if player.hand]
		return len(active_players) <= 1

	def play_turn(self, cards: List[Card]) -> bool:
		if not self.is_valid_turn(cards):
			print("Invalid turn played")
			return False
		self.last_played_cards = cards
		self.remove_played_cards_from_hand(cards)
		self.player_turn = self.next_player_turn()
		return True
		
	def is_valid_turn(self, cards: List[Card]) -> bool:
		return cards and 0 < len(cards) <= self.MAX_CARDS_PER_TURN and not self.is_last_player_with_a_hand()

	def remove_played_cards_from_hand(self, cards: List[Card]) -> None:
		current_player = self.get_current_player()
		for card in cards:
			current_player.hand.remove(card)

	def check_last_played_cards(self) -> bool:
		return all(card.rank == self.table_card or card.rank == 'Joker' for card in self.last_played_cards)


	def challenge_last_player(self) -> bool:
		if not self.last_played_cards:
			print("Invalid challenge played")
			return False
		self.round_finished = True
		if self.check_last_played_cards():
			self.lose_round(self.get_current_player())
		else:
			self.lose_round(self.players[self.previous_player_turn()])
		return True
	
	def lose_round(self, player: Player) -> None:
		print(f"Player {player.name} has lost! You will be shot now!")
		self.shoot_player(player)
	
	def shoot_player(self, player: Player) -> None:
		player.bullets_shot += 1
		if player.bullets_shot == player.death_bullet:
			self.remove_player(player)
			print(f"Player {player.name} has been shot")
			self.player_turn = 0
		else:
			self.player_turn = self.players.index(player)
	