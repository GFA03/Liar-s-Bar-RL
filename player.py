from abc import ABC, abstractmethod
from typing import List
from card import Card

class Player(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.hand: List[Card] = []
        self.bullets_shot: int = 0
        self.death_bullet: int = 0

    def take_shot(self) -> bool:
        """Handle a shot and return whether the player has reached their death bullet."""
        self.bullets_shot += 1
        return self.bullets_shot == self.death_bullet

    @abstractmethod
    def choose_action(self) -> tuple[int, List[Card]]:
        """
        Abstract method to choose an action.
        Returns a tuple, first being type of action, second(optional) being list of cards to play
        """
        pass

    @abstractmethod
    def choose_cards(self) -> List[Card]:
        """Abstract method to choose cards."""
        pass