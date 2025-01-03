from typing import List
from card import Card
from deck import Deck
import random

from player import Player

class InvalidTurnError(Exception):
	def __init__(self, message: str, player: Player, cards: List[Card]):
		super().__init__(f"{message} | Player: {player.name} | Cards: {cards}")

class InvalidChallengeError(Exception):
	pass

class LiarsBarGame:
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
		self.previous_player_index = None

	def add_player(self, player) -> None:
		self.players.append(player)

	def next_player_turn(self) -> int:
		next_index = (self.player_turn + 1) % len(self.players)
		while not self.players[next_index].hand:
			next_index = (next_index + 1) % len(self.players)
		return next_index

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

	def play_turn(self, cards: List[Card]) -> None:
		self.is_valid_turn(cards)
		self.last_played_cards = cards
		self.remove_played_cards_from_hand(cards)
		self.previous_player_index = self.player_turn
		self.player_turn = self.next_player_turn()
		
	def is_valid_turn(self, cards: List[Card]) -> bool:
		if cards is None or len(cards) <= 0 or len(cards) > self.MAX_CARDS_PER_TURN:
			raise InvalidTurnError("You must play between 1 and 3 cards", self.get_current_player(), cards)
		if self.is_last_player_with_a_hand():
			raise InvalidTurnError("You are the last player with cards! You must challenge!", self.get_current_player(), cards)
		if not self.check_player_has_given_cards(self.get_current_player(), cards):
			raise InvalidTurnError("You must play cards that you have in your hand", self.get_current_player(), cards)
	
	def check_player_has_given_cards(self, player: Player, cards: List[Card]) -> bool:
		return all(card in player.hand for card in cards)

	def remove_played_cards_from_hand(self, cards: List[Card]) -> None:
		current_player = self.get_current_player()
		for card in cards:
			current_player.hand.remove(card)

	def check_last_played_cards(self) -> bool:
		return all(card.rank == self.table_card or card.rank == 'Joker' for card in self.last_played_cards)


	def challenge_last_player(self) -> None:
		if not self.last_played_cards:
			raise InvalidChallengeError("No cards played yet")
		self.round_finished = True
		if self.check_last_played_cards():
			self.lose_round(self.get_current_player())
		else:
			self.lose_round(self.players[self.previous_player_index])
	
	def lose_round(self, player: Player) -> None:
		print(f"Player {player.name} has lost! You will be shot now!")
		self.shoot_player(player)
	
	def shoot_player(self, player: Player) -> None:
		if player.take_shot():
			self.remove_player(player)
			print(f"Player {player.name} has been killed")
		self.player_turn = 0 if player not in self.players else self.players.index(player)