from game.card import Card
import random

class Deck:
  def __init__(self) -> None:
    self.cards = []
    ranks = ['Q', 'K', 'A']
    for rank in ranks:
      for _ in range(6):
        self.cards.append(Card(rank))
    self.cards.append(Card('Joker'))
    self.cards.append(Card('Joker'))

  def shuffle(self) -> None:
    random.shuffle(self.cards)

  def draw(self) -> Card:
    return self.cards.pop()
  
  def __str__(self) -> str:
    return f"Deck of {len(self.cards)} cards"