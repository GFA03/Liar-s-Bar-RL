import os
import time
from typing import List
from card import Card
from game import InvalidChallengeError, InvalidTurnError, LiarsBarEnv
from player import Player

def read_int(statement: str = '') -> int:
  while True:
    try:
      return int(input(statement))
    except ValueError:
      print("Invalid input, try again")

def choose_action(current_player: Player) -> tuple[int, List[Card]]:
  chosen_option = -1
  while chosen_option not in [1, 2]:
    chosen_option = read_int("Choose option\n1 - Play hand\n2 - Challenge previous player\n")
  if chosen_option == 1:
    return (1, choose_cards(current_player))
  return (2, [])

def choose_cards(current_player: Player) -> List[Card]:
  chosen_cards = []
  display_hand(current_player)
  while True:
    print("Chosen cards:\n", *chosen_cards)
    chosen_card = read_int("Type the number of the card you want to play, or -1 to stop\n")
    if chosen_card == -1:
      break
    if 0 <= chosen_card < len(current_player.hand):
      toggle_card_selection(current_player, chosen_cards, chosen_card)
  return chosen_cards

def display_hand(player: Player) -> None:
  for i, card in enumerate(player.hand):
    print(f"{i} - {card}")

def toggle_card_selection(current_player: Player, chosen_cards: List[Card], chosen_card: int) -> None:
  card = current_player.hand[chosen_card]
  if card in chosen_cards:
      chosen_cards.remove(card)
  else:
      chosen_cards.append(card)

def initialize_game() -> LiarsBarEnv:
  game = LiarsBarEnv()
  game.add_player(Player("Alex"))
  game.add_player(Player("Edi"))
  return game

def play_round(game: LiarsBarEnv) -> None:
  display_round_status(game)
  game.initialize_round()
  while not game.round_finished:
    print(f"Current player: {game.get_current_player().name}")
    print("Current table card:", game.table_card)
    execute_player_action(game)
    time.sleep(2)
    os.system('cls' if os.name == 'nt' else 'clear')

def display_round_status(game: LiarsBarEnv) -> None:
  print('Initializing round!')
  print('Players:')
  for player in game.players:
    print(f'{player.name} - {player.bullets_shot}/{game.MAX_DEATH_BULLET} shots')
  print()

def execute_player_action(game: LiarsBarEnv) -> None:
  while True:
    chosen_option, chosen_cards = choose_action(game.get_current_player())
    try:
      if chosen_option == 1:
        game.play_turn(chosen_cards)
      else:
        game.challenge_last_player()
      break
    except (InvalidTurnError, InvalidChallengeError) as e:
      print(e)
    time.sleep(1)

if __name__ == '__main__':
  game = initialize_game()
  while len(game.players) > 1:
    play_round(game)
    time.sleep(1)