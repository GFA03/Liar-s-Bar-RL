import pytest
from env import LiarsBarEnv, Card  # Assuming this is saved as liars_bar_env.py

INITIAL_HAND_SIZE = 5

@pytest.fixture
def env():
    """Fixture to create and reset the environment before each test."""
    environment = LiarsBarEnv(num_players=2)
    environment.reset()
    return environment

def test_action_space(env):
    expected_size = 2 ** INITIAL_HAND_SIZE
    assert env.action_space.n == expected_size, "Action space size mismatch."

def test_observation_space(env):
    obs_space = env.observation_space
    assert "hand" in obs_space.spaces, "Observation space should include 'hand'."
    assert "table_card" in obs_space.spaces, "Observation space should include 'table_card'."
    assert "last_played" in obs_space.spaces, "Observation space should include 'last_played'."
    assert "player_turn" in obs_space.spaces, "Observation space should include 'player_turn'."

def test_action_decoding():
    action = 18  # Binary: 10010
    expected_cards = [1 if i in [0, 3] else 0 for i in range(INITIAL_HAND_SIZE)]
    decoded_action = [int(x) for x in format(action, f"0{INITIAL_HAND_SIZE}b")]
    assert decoded_action == expected_cards, "Action decoding failed."

def test_get_available_actions(env):
    env.players[0]["hand"] = [Card.Q, Card.K, Card.A, Card.Q, Card.Joker]
    env.player_turn = 0

    # Play the Joker card
    first_action = 1
    env.step(first_action)

    # Get available actions for second player
    available_actions = env._get_available_actions()
    # Challenge action should always be available
    challenge_action = 0
    assert challenge_action in available_actions, "Challenge action missing."

    # Play action: Binary 10010 (indices 1 and 4)
    expected_play_action = 18
    assert expected_play_action in available_actions, "Valid play action missing."

    env.step(expected_play_action)

    available_actions = env._get_available_actions()
    assert first_action not in available_actions, "Invalid action is valid"



def test_play_turn(env):
    env.players[0]["hand"] = [Card.Q, Card.K, Card.A, Card.Q, Card.Joker]
    env.player_turn = 0

    # Play action: Binary 10010 (indices 1 and 4)
    play_action = 18
    env.step(play_action)

    # Check if cards at indices 1 and 4 are marked as played
    expected_hand = [-1, Card.K, Card.A, -1, Card.Joker]
    assert env.players[0]["hand"] == expected_hand, "Cards were not played correctly."

def test_challenge_action(env):
    env.players[0]["hand"] = [Card.Q, Card.K, Card.A, Card.Q, Card.Joker]
    env.table_card = Card.K
    env.player_turn = 0
    # 16 - 10000
    env.step(16)

    # Challenge action
    env._challenge_previous_player()

    # As the last played cards are a bluff, Player 0 gets shot
    assert env.players[0]["bullets_shot"] > 0, "Challenge did not penalize correctly."

def test_reset(env):
    env.reset()
    assert len(env.players) == 2, "Player count mismatch after reset."
    assert env.table_card in [Card.Q, Card.K, Card.A], "Table card should be initialized after reset."
    assert env.round_finished is False
    assert env.player_turn == 0
    assert env.last_played_cards == []
