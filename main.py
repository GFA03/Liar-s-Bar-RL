import os
import time
from typing import List
from card import Card
from game import InvalidChallengeError, InvalidTurnError, LiarsBarEnv
from player import Player
from utils import read_int

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
    clear_screen()

def clear_screen() -> None:
  os.system('cls' if os.name == 'nt' else 'clear')

def display_round_status(game: LiarsBarEnv) -> None:
  print('Initializing round!')
  print('Players:')
  for player in game.players:
    print(f'{player.name} - {player.bullets_shot}/{game.MAX_DEATH_BULLET} shots')
  print()

def execute_player_action(game: LiarsBarEnv) -> None:
  while True:
    chosen_option, chosen_cards = game.get_current_player().choose_action()
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