class Card:
  def __init__(self, rank: str):
    self.rank = rank
  
  def __str__(self):
    return f"{self.rank}\n"