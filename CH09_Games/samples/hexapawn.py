from ea_games import HexapawnGameController
from easyAI import AI_Player, Human_Player, Negamax


if __name__ == '__main__':
    # Compute the score
    scoring = lambda game: -100 if game.loss_condition() else 0

    # Define the algorithm
    # algorithm = Negamax(12, scoring)
    algorithm = Negamax(13, scoring)

    # Start the game
    game = HexapawnGameController([AI_Player(algorithm),  AI_Player(algorithm)])
    game.play()
    print('\nPlayer', game.nopponent, 'wins after', game.nmove, 'turns')
