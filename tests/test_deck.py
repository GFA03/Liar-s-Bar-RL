from deck import Deck

def test_deck_has_52_cards():
    deck = Deck()
    assert len(deck.cards) == 52

def test_deck_has_4_of_each_rank():
    deck = Deck()
    for rank in range(1, 14):
        assert len([card for card in deck.cards if card.rank == rank]) == 4

def test_deck_has_13_of_each_suit():
    deck = Deck()
    for suit in ['Hearts', 'Diamonds', 'Clubs', 'Spades']:
        assert len([card for card in deck.cards if card.suit == suit]) == 13

def test_deck_has_no_duplicates():
    deck = Deck()
    assert len(set(deck.cards)) == 52