import numpy as np
from easyAI import TwoPlayersGame


class LastCoinStanding(TwoPlayersGame):
    def __init__(self, players):
        # Define the players. Necessary parameter.
        self.players = players

        # Define who starts the game. Necessary parameter.
        self.nplayer = 1

        # Overall number of coins in the pile
        self.num_coins = 25

        # Define max number of coins per move
        self.max_coins = 4

    # Define possible moves
    def possible_moves(self):
        return [str(x) for x in range(1, self.max_coins + 1)]

    # Remove coins
    def make_move(self, move):
        self.num_coins -= int(move)

    # Did the opponent take the last coin?
    def win(self):
        return self.num_coins <= 0

    # Stop the game when somebody wins
    def is_over(self):
        return self.win()

    # Compute score
    def scoring(self):
        return 100 if self.win() else 0

    # Show number of coins remaining in the pile
    def show(self):
        print(self.num_coins, 'coins left in the pile')


class ConnectFourGameController(TwoPlayersGame):
    def __init__(self, players, board=None):
        # Define the players
        self.players = players

        # Define the configuration of the board
        self.board = board if (board != None) else (np.array([[0 for i in range(7)] for j in range(6)]))

        # Define who starts the game
        self.nplayer = 1

        # Define the positions
        self.pos_dir = np.array([[[i, 0], [0, 1]] for i in range(6)] +
                                [[[0, i], [1, 0]] for i in range(7)] +
                                [[[i, 0], [1, 1]] for i in range(1, 3)] +
                                [[[0, i], [1, 1]] for i in range(4)] +
                                [[[i, 6], [1, -1]] for i in range(1, 3)] +
                                [[[0, i], [1, -1]] for i in range(3, 7)])

    # Define possible moves
    def possible_moves(self):
        return [i for i in range(7) if (self.board[:, i].min() == 0)]

    # Define how to make the move
    def make_move(self, column):
        line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.nplayer

    # Show the current status
    def show(self):
        print('\n' + '\n'.join(
            ['0 1 2 3 4 5 6', 13 * '-'] +
            [' '.join([['.', 'O', 'X'][self.board[5 - j][i]]
                       for i in range(7)]) for j in range(6)]))

    # Define what a loss_condition looks like
    def loss_condition(self):
        for pos, direction in self.pos_dir:
            streak = 0
            while (0 <= pos[0] <= 5) and (0 <= pos[1] <= 6):
                if self.board[pos[0], pos[1]] == self.nopponent:
                    streak += 1
                    if streak == 4:
                        return True
                else:
                    streak = 0

                pos = pos + direction

        return False

    # Check if the game is over
    def is_over(self):
        return (self.board.min() > 0) or self.loss_condition()

    # Compute the score
    def scoring(self):
        return -100 if self.loss_condition() else 0


class HexapawnGameController(TwoPlayersGame):
    def __init__(self, players, size=(4, 4)):
        self.size = size
        num_pawns, len_board = size
        p = [[(i, j) for j in range(len_board)]
             for i in [0, num_pawns - 1]]

        for i, d, goal, pawns in [(0, 1, num_pawns - 1,
                                   p[0]), (1, -1, 0, p[1])]:
            players[i].direction = d
            players[i].goal_line = goal
            players[i].pawns = pawns

        # Define the players
        self.players = players

        # Define who starts first
        self.nplayer = 1

        # Define the alphabets
        self.alphabets = 'ABCDEFGHIJ'

        # Convert B4 to (1, 3)
        self.to_tuple = lambda s: (self.alphabets.index(s[0]),
                                   int(s[1:]) - 1)

        # Convert (1, 3) to B4
        self.to_string = lambda move: ' '.join([self.alphabets[
            move[i][0]] + str(move[i][1] + 1)
            for i in (0, 1)])

    # Define the possible moves
    def possible_moves(self):
        moves = []
        opponent_pawns = self.opponent.pawns
        d = self.player.direction

        for i, j in self.player.pawns:
            if (i + d, j) not in opponent_pawns:
                moves.append(((i, j), (i + d, j)))

            if (i + d, j + 1) in opponent_pawns:
                moves.append(((i, j), (i + d, j + 1)))

            if (i + d, j - 1) in opponent_pawns:
                moves.append(((i, j), (i + d, j - 1)))

        return list(map(self.to_string, [(i, j) for i, j in moves]))

    # Define how to make a move
    def make_move(self, move):
        move = list(map(self.to_tuple, move.split(' ')))
        ind = self.player.pawns.index(move[0])
        self.player.pawns[ind] = move[1]

        if move[1] in self.opponent.pawns:
            self.opponent.pawns.remove(move[1])

    # Define what a loss looks like
    def loss_condition(self):
        return (any([i == self.opponent.goal_line
                for i, j in self.opponent.pawns])
                or (self.possible_moves() == []))

    # Check if the game is over
    def is_over(self):
        return self.loss_condition()

    # Show the current status
    def show(self):
        def f(x): return '1' if x in self.players[0].pawns else (
            '2' if x in self.players[1].pawns else '.')

        print("\n".join([" ".join([f((i, j))
                                   for j in range(self.size[1])])
                         for i in range(self.size[0])]))


class TictactoeGameController(TwoPlayersGame):
    def __init__(self, players):
        # Define the players
        self.players = players

        # Define who starts the game
        self.nplayer = 1 

        # Define the board
        self.board = [0] * 9
    
    # Define possible moves
    def possible_moves(self):
        return [a + 1 for a, b in enumerate(self.board) if b == 0]
    
    # Make a move
    def make_move(self, move):
        self.board[int(move) - 1] = self.nplayer

    # Does the opponent have three in a line?
    def loss_condition(self):
        possible_combinations = [[1,2,3], [4,5,6], [7,8,9],
            [1,4,7], [2,5,8], [3,6,9], [1,5,9], [3,5,7]]

        return any([all([(self.board[i-1] == self.nopponent)
                for i in combination]) for combination in possible_combinations]) 
        
    # Check if the game is over
    def is_over(self):
        return (self.possible_moves() == []) or self.loss_condition()
        
    # Show current position
    def show(self):
        print('\n'+'\n'.join([' '.join([['.', 'O', 'X'][self.board[3*j + i]]
                for i in range(3)]) for j in range(3)]))
                 
    # Compute the score
    def scoring(self):
        return -100 if self.loss_condition() else 0