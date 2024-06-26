from ea_games import ConnectFourGameController
from easyAI import (
    AI_Player, Human_Player, 
    Negamax, SSS
)


if __name__ == '__main__':
    # Define the algorithms that will be used
    # algo_neg = Negamax(5)
    algo_neg = Negamax(3)
    algo_sss = SSS(5)

    # Start the game
    game = ConnectFourGameController([AI_Player(algo_neg), AI_Player(algo_sss)])
    game.play()

    # Print the result
    if game.loss_condition():
        print('\nPlayer', game.nopponent, 'wins.')
    else:
        print("\nIt's a draw.")