from deck import Deck

def test_deck_has_14_cards():
    deck = Deck()
    assert len(deck.cards) == 14

def test_deck_has_4_of_each_rank():
    deck = Deck()
    for rank in ['Q', 'K', 'A']:
        assert len([card for card in deck.cards if card.rank == rank]) == 4

def test_deck_has_no_duplicates():
    deck = Deck()
    assert len(set(deck.cards)) == 14