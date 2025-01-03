import random
from typing import List
from game.card import Card
from game.player import Player

class RandomPlayer(Player):
  def choose_action(self) -> tuple[int, List[Card]]:
      action = random.choice([1, 2])
      if action == 1:
          return (1, self.choose_cards())
      return (2, [])
  
  def choose_cards(self) -> List[Card]:
      chosen_cards = []
      possible_cards = self.hand.copy()
      num_of_cards = random.randint(1, 3)
      for _ in range(min(num_of_cards, len(possible_cards))):
          chosen_card = random.choice(possible_cards)
          possible_cards.remove(chosen_card)
          chosen_cards.append(chosen_card)
      print(f"{self.name} chose cards: ", *chosen_cards)
      return chosen_cards
