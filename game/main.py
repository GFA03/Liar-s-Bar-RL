import os
import time
from game.game import InvalidChallengeError, InvalidTurnError, LiarsBarGame
from game.human_player import HumanPlayer
from game.random_player import RandomPlayer

def initialize_game() -> LiarsBarGame:
  game = LiarsBarGame()
  while not add_players(game):
    print("You should add at least two players")
    pass
  return game

def add_players(game: LiarsBarGame) -> None:
  name = input("Enter a name or press Enter to stop: ")
  while name != "" and len(game.players) < 3:
    game.add_player(HumanPlayer(name))
    name = input("Enter a name or press Enter to stop: ")
  if name != "":
    game.add_player(HumanPlayer(name))
  return 1 < len(game.players) <= 4

def play_round(game: LiarsBarGame) -> None:
  display_round_status(game)
  game.initialize_round()
  while not game.round_finished:
    if game.last_played_cards is not None:
      print(f"Last player has played {len(game.last_played_cards)} cards")
    print(f"Current player: {game.get_current_player().name}")
    print("Current table card:", game.table_card)
    execute_player_action(game)
    input("Press Enter to continue...")
    clear_screen()

def clear_screen() -> None:
  os.system('cls' if os.name == 'nt' else 'clear')

def display_round_status(game: LiarsBarGame) -> None:
  print('Initializing round!')
  print('Players:')
  for player in game.players:
    print(f'{player.name} - {player.bullets_shot}/{game.MAX_DEATH_BULLET} shots')
  print()

def execute_player_action(game: LiarsBarGame) -> None:
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