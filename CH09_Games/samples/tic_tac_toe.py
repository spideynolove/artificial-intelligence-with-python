from ea_games import TictactoeGameController
from easyAI import AI_Player, Negamax
from easyAI.Player import Human_Player


if __name__ == "__main__":
    # Define the algorithm
    algorithm = Negamax(7)

    # Start the game
    # TictactoeGameController([Human_Player(), AI_Player(algorithm)]).play()
    TictactoeGameController([AI_Player(algorithm), AI_Player(algorithm)]).play()