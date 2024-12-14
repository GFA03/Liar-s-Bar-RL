import time
from typing import List
from card import Card
from game import LiarsBarEnv
from player import Player

def read_int(statement: str = '') -> int:
  while True:
    try:
      return int(input(statement))
    except ValueError:
      print("Invalid input, try again")

'''
Returns the chosen option, and a list of cards for the first option
'''
def choose_action(current_player: Player) -> tuple[int, List[Card]]:
  chosen_option = -1
  while chosen_option != 1 and chosen_option != 2:
    chosen_option = read_int("Choose option\n1 - Play hand\n2 - Challenge previous player\n")
  if chosen_option == 1:
    chosen_cards = []
    while len(chosen_cards) == 0 or len(chosen_cards) > 3:
      print("Choose cards to play (up to 3)")
      chosen_cards = choose_cards(current_player)
    return (1, chosen_cards)
  return (2, [])

def choose_cards(current_player: Player) -> List[Card]:
  chosen_cards = []
  for i, card in enumerate(current_player.hand):
    print(f"{i} - {card}")
  print("Type the number of the card you want to play, or -1 to stop")
  while True:
    print("Chosen cards:\n", *chosen_cards)
    chosen_card = -2
    while chosen_card < -1 or chosen_card >= len(current_player.hand):
      chosen_card = read_int()
    if chosen_card == -1:
      break
    if current_player.hand[chosen_card] in chosen_cards:
      chosen_cards.remove(current_player.hand[chosen_card])
    else:
      chosen_cards.append(current_player.hand[chosen_card])
  return chosen_cards

if __name__ == '__main__':
  game = LiarsBarEnv()
  game.add_player(Player("Alex"))
  game.add_player(Player("Edi"))
  while len(game.players) > 1:
    print("Initializing round!")
    game.initialize_round()
    while not game.round_finished:
      print(f"Current player: {game.get_current_player().name}")
      print("Current table card:", game.table_card)
      option_status = False
      while option_status is False:
        chosen_option, chosen_cards = choose_action(game.get_current_player())
        if chosen_option == 1:
          option_status = game.play_turn(chosen_cards)
        else:
          option_status = game.challenge_last_player()
        time.sleep(1)
    time.sleep(2)
