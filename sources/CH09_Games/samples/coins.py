from ea_games import LastCoinStanding
from easyAI import id_solve, Human_Player, AI_Player
from easyAI.AI import TT


if __name__ == "__main__":
    # Define the transposition table
    tt = TT()

    # Define the method
    LastCoinStanding.ttentry = lambda self: self.num_coins

    # Solve the game
    result, depth, move = id_solve(LastCoinStanding,
                                   range(2, 20), win_score=100, tt=tt)
    print(result, depth, move)

    # Start the game
    # game = LastCoinStanding([AI_Player(tt), Human_Player()])
    game = LastCoinStanding([AI_Player(tt), AI_Player(tt)])
    game.play()