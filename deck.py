from card import Card
import random

class Deck:
  def __init__(self):
    self.cards = []
    for suit in ["Hearts", "Diamonds", "Clubs", "Spades"]:
      for rank in range(1, 14):
        self.cards.append(Card(rank, suit))

  def shuffle(self):
    random.shuffle(self.cards)

  def draw(self):
    return self.cards.pop()
  
  def __str__(self):
    return f"Deck of {len(self.cards)} cards"