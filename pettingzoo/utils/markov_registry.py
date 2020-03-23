"""Registry of games, that can be played as markov game with RlLib"""


def _import_prison():
    from pettingzoo.gamma import prison
    return prison


def _import_simple_spread():
    from pettingzoo.mpe import simple_spread
    return simple_spread


MARKOV_GAMES = {
    "prison": _import_prison,
    'simple_spread': _import_simple_spread,
}


def get_game_class(game):
    """Returns the class of a known agent given its name."""

    try:
        return _get_game_class(game)
    except ImportError:
        print('Game not listed in Markov games registry')


def _get_game_class(game):
    if game in MARKOV_GAMES:
        return MARKOV_GAMES[game]()
    else:
        raise Exception(("Unknown algorithm {}.").format(game))
