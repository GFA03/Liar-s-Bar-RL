from typing import List
from card import Card
from player import Player
from utils import read_int

class HumanPlayer(Player):      
  def choose_action(self) -> tuple[int, List[Card]]:
      chosen_option = -1
      while chosen_option not in [1, 2]:
          chosen_option = read_int(f"{self.name}, choose option\n1 - Play hand\n2 - Challenge previous player\n")
      if chosen_option == 1:
          return (1, self.choose_cards())
      return (2, [])
  
  def choose_cards(self) -> List[Card]:
      chosen_cards = []
      self.display_hand()
      while True:
          print("Chosen cards:\n", *chosen_cards)
          chosen_card = read_int("Type the number of the card you want to play, or -1 to stop\n")
          if chosen_card == -1:
              break
          if 0 <= chosen_card < len(self.hand):
              self.toggle_card_selection(chosen_cards, chosen_card)
          else:
              print("Invalid card number, try again")
      return chosen_cards
  
  def display_hand(self) -> None:
      for i, card in enumerate(self.hand):
          print(f"{i} - {card}")

  def toggle_card_selection(self, chosen_cards: List[Card], chosen_card: int) -> None:
      card = self.hand[chosen_card]
      if card in chosen_cards:
          chosen_cards.remove(card)
      else:
          chosen_cards.append(card)
