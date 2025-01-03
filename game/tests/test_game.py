from game.card import Card
from game.game import InvalidChallengeError, InvalidTurnError, LiarsBarGame
from game.human_player import HumanPlayer
import pytest

def test_initialize_round():
  game = LiarsBarGame()
  game.add_player(HumanPlayer("Player A"))
  game.add_player(HumanPlayer("Player B"))
  game.initialize_round()
  for player in game.players:
    assert len(player.hand) == 5
    assert player.bullets_shot == 0
    assert player.death_bullet >= 1 and player.death_bullet <= 6
  assert game.table_card in LiarsBarGame.TABLE_CARDS
  assert game.last_played_cards == None
  assert game.round_finished == False

def test_next_player_turn():
  game = LiarsBarGame()
  game.add_player(HumanPlayer("Player A"))
  game.add_player(HumanPlayer("Player B"))
  game.initialize_round()
  assert game.next_player_turn() == 1
  game.player_turn = 1
  assert game.next_player_turn() == 0

def test_previous_player_turn():
  game = LiarsBarGame()
  player_a = HumanPlayer("Player A")
  player_b = HumanPlayer("Player B")
  game.add_player(player_a)
  game.add_player(player_b)
  game.initialize_round()
  game.play_turn([player_a.hand[0], player_a.hand[1], player_a.hand[2]])
  assert game.previous_player_index == 0
  game.play_turn([player_b.hand[0]])
  assert game.previous_player_index == 1

def test_get_current_player():
  game = LiarsBarGame()
  game.add_player(HumanPlayer("Player A"))
  game.add_player(HumanPlayer("Player B"))
  assert game.get_current_player().name == "Player A"
  game.player_turn = 1
  assert game.get_current_player().name == "Player B"

def test_remove_player():
  game = LiarsBarGame()
  player_a = HumanPlayer("Player A")
  player_b = HumanPlayer("Player B")
  game.add_player(player_a)
  game.add_player(player_b)
  game.remove_player(player_a)
  assert len(game.players) == 1
  assert game.players[0].name == "Player B"


def test_invalid_turn():
  game = LiarsBarGame()
  game.add_player(HumanPlayer("Player A"))
  game.add_player(HumanPlayer("Player B"))
  game.initialize_round()
  with pytest.raises(InvalidTurnError):
      game.play_turn([])  # Playing 0 cards should raise an error
  with pytest.raises(InvalidTurnError):
      game.play_turn([game.get_current_player().hand[0]] * 4)  # Playing 4 cards should raise an error


def test_play_turn():
  game = LiarsBarGame()
  game.add_player(HumanPlayer("Player A"))
  game.add_player(HumanPlayer("Player B"))
  game.initialize_round()
  current_player: HumanPlayer = game.get_current_player()
  played_cards = [current_player.hand[0], current_player.hand[1]]
  remaining_cards = [current_player.hand[2], current_player.hand[3], current_player.hand[4]]
  game.play_turn(played_cards)
  assert game.last_played_cards == played_cards
  assert current_player.hand == remaining_cards
  assert game.player_turn == 1

def test_challenge_on_start_round():
  game = LiarsBarGame()
  game.add_player(HumanPlayer("Player A"))
  game.add_player(HumanPlayer("Player B"))
  game.initialize_round()
  with pytest.raises(InvalidChallengeError):
    game.challenge_last_player()

def test_play_turn_on_last_player():
  game = LiarsBarGame()
  player_a = HumanPlayer("Player A")
  player_b = HumanPlayer("Player B")
  game.add_player(player_a)
  game.add_player(player_b)
  game.initialize_round()
  game.play_turn([player_a.hand[0], player_a.hand[1], player_a.hand[2]])
  game.play_turn([player_b.hand[0]])
  game.play_turn([player_a.hand[0], player_a.hand[1]])
  # Player B is the last player so he can only challenge
  with pytest.raises(InvalidTurnError):
    game.play_turn([player_b.hand[0]])
  game.challenge_last_player()

def test_challenge_last_player_liar():
  game = LiarsBarGame()
  player_a = HumanPlayer("Player A")
  player_b = HumanPlayer("Player B")
  game.add_player(player_a)
  game.add_player(player_b)
  game.initialize_round()

  # Make sure Player A doesn't die
  player_a.death_bullet = 6

  # Set the played card to be a lie
  player_a.hand[0].rank = 'A' if game.table_card == 'J' else 'J'
  game.play_turn([player_a.hand[0]])

  # Player B challenges Player A
  game.challenge_last_player()
  assert player_a.bullets_shot == 1
  assert game.player_turn == 0
  
def test_challenge_last_player_truth_on_rank():
  game = LiarsBarGame()
  player_a = HumanPlayer("Player A")
  player_b = HumanPlayer("Player B")
  game.add_player(player_a)
  game.add_player(player_b)
  game.initialize_round()

  # Make sure player B doesn't die
  player_b.death_bullet = 6

  # Set the played card to be the table card
  player_a.hand[0].rank = game.table_card
  game.play_turn([player_a.hand[0]])

  # Player B challenges Player A and fails
  game.challenge_last_player()
  assert player_b.bullets_shot == 1
  # Player B starts the next round as he is the loser
  assert game.player_turn == 1

def test_challenge_last_player_truth_on_joker():
  game = LiarsBarGame()
  player_a = HumanPlayer("Player A")
  player_b = HumanPlayer("Player B")
  game.add_player(player_a)
  game.add_player(player_b)
  game.initialize_round()

  # Make sure player B doesn't die
  player_b.death_bullet = 6

  player_a.hand[0].rank = 'Joker'
  game.play_turn([player_a.hand[0]])
  game.challenge_last_player()
  assert player_b.bullets_shot == 1
  assert game.player_turn == 1

def test_is_last_player_with_a_hand():
    game = LiarsBarGame()
    player_a = HumanPlayer("Player A")
    player_b = HumanPlayer("Player B")
    game.add_player(player_a)
    game.add_player(player_b)
    game.initialize_round()
    player_a.hand = []
    assert game.is_last_player_with_a_hand() == True
    player_a.hand = [Card('Q')]
    assert game.is_last_player_with_a_hand() == False