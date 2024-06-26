# Game with AI

- Using **search algorithms** in games
- **Combinatorial** search
- **Minimax** algorithm
- **Alpha-Beta** pruning
- **Negamax** algorithm
- Building a bot to play **Last Coin Standing**
- Building a bot to play **Tic Tac Toe**
- Building two bots to play **Connect Four** against each other
- Building two bots to play **Hexapawn** against each other

## Table of Contents

- [Game with AI](#game-with-ai)
  - [Table of Contents](#table-of-contents)
  - [Using search algorithms in games](#using-search-algorithms-in-games)
    - [GPT](#gpt)
    - [Book](#book)
  - [Combinatorial search](#combinatorial-search)
    - [GPT](#gpt-1)
    - [Book](#book-1)
  - [Minimax algorithm](#minimax-algorithm)
    - [GPT](#gpt-2)
    - [Book](#book-2)
  - [Alpha-Beta pruning](#alpha-beta-pruning)
  - [Negamax algorithm](#negamax-algorithm)
  - [easyAI library](#easyai-library)
  - [Last Coin Standing](#last-coin-standing)
  - [Tic-Tac-Toe](#tic-tac-toe)
  - [Connect Four](#connect-four)
  - [Hexapawn](#hexapawn)
  - [Summary](#summary)

## Using search algorithms in games

### GPT

- Search algorithms are widely used in game development for various purposes, including pathfinding, decision-making, and game state evaluation. 

**1. Pathfinding:**

   - **A* Search:** A* is a popular search algorithm for finding the shortest path from a start point to a goal point while avoiding obstacles. It's commonly used in 2D and 3D games for character movement and navigation.

**2. Game Trees and Decision-Making:**
   - **Minimax Algorithm:** Minimax is used in turn-based games, such as chess, to make optimal decisions by considering all possible moves and their outcomes.
   - **Alpha-Beta Pruning:** Alpha-Beta pruning is an enhancement of the minimax algorithm, optimizing the search process by eliminating branches that can't lead to better outcomes.
   - **Monte Carlo Tree Search (MCTS):** MCTS is used in games with complex decision spaces, like Go and Poker. It builds a tree of possible game states and selects the best moves based on simulations.

**3. Game State Evaluation:**
   - **Heuristic Search:** Heuristic search algorithms, like A* with heuristic functions, are used to evaluate game states in real-time strategy games or puzzle games. They help AI agents make decisions.

**4. Pathfinding and Crowd Simulation:**
   - **Flocking Algorithms:** Flocking algorithms, like Reynolds' Boids, simulate the movement of groups of entities in games, such as schools of fish or flocks of birds.
   - **Dijkstra's Algorithm:** Dijkstra's algorithm is used in games for real-time pathfinding and routing, ensuring efficient and collision-free movement.

**5. Puzzle Solving:**
   - **Sudoku Solver:** Search algorithms can be used to solve puzzle games like Sudoku or crossword puzzles.

**6. Procedural Content Generation:**
   - **Cellular Automata:** Cellular automata are used to generate procedural content in games, including terrain, maps, and level design.

**7. Rule-Based Systems:**
   - **Expert Systems:** Rule-based systems use logical rules to guide game AI behavior, making decisions based on a set of predefined rules.

**8. Dynamic Difficulty Adjustment:**
   - Search algorithms can be employed to dynamically adjust the difficulty of a game based on the player's skill level and performance.

**9. State Space Exploration:**
   - In complex strategy games, AI agents may explore the state space using search algorithms to devise long-term plans and strategies.

**10. Puzzle Solving and Hidden Object Games:**
    - For puzzle-solving games and hidden object games, search algorithms can be used to guide players in finding items or solving puzzles.

### Book

- used in games to figure out a **strategy**.
- search through the **possibilities** and pick the **best move**.
- various parameters to think about: speed, accuracy, complexity
- algorithms consider **all possible** actions
- find the **optimal set of moves**
- algorithms use **winning conditions** to find the set of move
- **constantly re-evaluate** after each move.

## Combinatorial search

### GPT

- A category of search algorithms used in artificial intelligence and computer science to explore and find solutions within a discrete, often finite, search space. 
- It's primarily employed in problems where the goal is to find a combination or arrangement of items, elements, or variables that satisfies certain constraints or criteria. 
- Combinatorial search problems are common in various fields, including computer science, operations research, logistics, and artificial intelligence. 

**1. State Space:** Combinatorial search problems can often be framed as a state space, where each state represents a specific configuration or combination of elements. The goal is to navigate through this state space to find a solution or optimal combination.

**2. Search Tree:** In many combinatorial search problems, a search tree is constructed, where each node represents a state, and edges represent transitions from one state to another through various actions or decisions.

**3. Branch and Bound:** Branch and bound is a technique used to systematically explore the search space by dividing it into smaller subproblems, bounding the potential solutions, and prioritizing the exploration of promising subspaces.

**4. Pruning:** Pruning involves eliminating parts of the search space that cannot lead to a solution or an improved solution. Common pruning techniques include heuristic-based pruning and constraint propagation.

**5. Backtracking:** Backtracking is a common algorithmic technique for solving combinatorial search problems. It involves exploring potential solutions incrementally and undoing choices if they lead to a dead-end, thereby "backtracking" to a previous state.

**6. Heuristic Functions:** Heuristic functions are used to estimate the potential of a state to lead to a solution. These functions guide the search process by prioritizing states that are more likely to yield good results.

**7. Constraint Satisfaction:** Constraint satisfaction problems involve finding a combination of values for a set of variables that satisfies a given set of constraints. Backtracking and constraint propagation techniques are often used in these problems.

**8. Combinatorial Optimization:** Combinatorial optimization problems seek to find the best possible solution from a finite set of possibilities. These problems include the traveling salesman problem, knapsack problem, and graph coloring.

**9. Combinatorial Explosion:** Many combinatorial search problems exhibit combinatorial explosion, meaning the number of potential combinations grows exponentially with the problem size. Efficient search techniques are required to address this challenge.

**10. Applications:** Combinatorial search is applied in a wide range of applications, including route planning, scheduling, game playing, network design, cryptography, and artificial intelligence.

### Book

- Search algorithms drawback is exhaustive search space? ~ brute force search
  - explores the entire search space and tests every possible solution
  - in the worst case, we will have to explore all the possible solutions before we get the right solution.
  - enormous posible solutions if game get more complex ~ expensive computating
- combinatorial search is solving these problems.
  - where search algorithms efficiently??
  - using **heuristics** or by **reducing** the size of the search space.
  - using **pruning** strategies.
  
## Minimax algorithm

### GPT

- The primary objective of the Minimax algorithm is to determine the optimal move for a player by considering all possible moves and their outcomes, while assuming that the opponent will also make the best moves to **minimize the player's gain**. 
- The algorithm is based on the principle of **minimizing the maximum possible loss**.

**Key Concepts:**

1. **Two-Player, Zero-Sum Games:** Minimax is used in games where two players take turns, and the outcome is zero-sum, meaning that one player's gain is equivalent to the other player's loss. The objective is to maximize one's own gain while minimizing the opponent's.

2. **Game Tree:** The game's potential moves and states are represented as a tree, with each node in the tree representing a game state and the edges representing possible moves or transitions between states.

3. **Maximizer and Minimizer:** In the Minimax algorithm, one player is considered the maximizer, whose goal is to maximize the score (usually their own) at each level of the tree. The other player is the minimizer, whose goal is to minimize the score (usually the maximizer's) at each level.

**Algorithm Steps:**

1. **Recursive Search:** The Minimax algorithm involves a recursive search through the game tree, starting at the current state (the root of the tree).

2. **Depth-First Search:** The algorithm explores the tree in a depth-first manner, evaluating each node at the current level before moving on to the next level.

3. **Terminal Nodes:** When a terminal node is reached (representing the end of the game or a game state where no further moves are possible), the algorithm assigns a value to that node based on the outcome of the game.

4. **Backtracking:** As the algorithm backtracks from terminal nodes to higher levels, it propagates values up the tree. At each level, the maximizer seeks the maximum value, while the minimizer seeks the minimum value.

5. **MinMax Function:** The core of the algorithm is the MinMax function. It alternates between maximizing and minimizing values, computing the best move for the current player based on the available moves and their associated values.

6. **Pruning (Alpha-Beta Pruning):** To improve the efficiency of the search, the algorithm often incorporates alpha-beta pruning. This technique eliminates branches of the tree that can't lead to a better result, reducing the number of nodes to evaluate.

**Pseudocode for Minimax:**

Here is a simplified pseudocode representation of the Minimax algorithm:

```plaintext
function minimax(node, depth, maximizingPlayer):
    if depth is 0 or node is a terminal node:
        return the heuristic value of the node
    
    if maximizingPlayer:
        bestValue = -∞
        for child in node's children:
            value = minimax(child, depth - 1, False)
            bestValue = max(bestValue, value)
        return bestValue
    else:
        bestValue = +∞
        for child in node's children:
            value = minimax(child, depth - 1, True)
            bestValue = min(bestValue, value)
        return bestValue
```

### Book

- One such strategy used by combinatorial search. 
- When two players are playing against each other, they are basically working towards **opposite** goals. 
- So each side needs to predict **what** the opposing player is **going to do in order to win** the game. 
- It will try to **minimize the function that the opponent is trying to maximize**.
- The computer can only optimize the moves based on the current state using a heuristic
- The computer **constructs a tree** and it starts from the bottom. It evaluates which moves would benefit its opponent.
- opponent will make the moves that would **benefit them the most**, this outcome is one of the terminal nodes of the tree and the **computer uses this position to work backwards**
- **Each** option that's available to the computer can be **assigned a value** and it can then **pick the highest value** to take an **action**.

## Alpha-Beta pruning

- A powerful optimization technique used in conjunction with the Minimax algorithm to **reduce the number of nodes that need to be evaluated in a game tree**. 
- It's particularly effective in two-player, zero-sum games, such as chess and checkers, where the goal is to find the best move while **considering the opponent's best responses**.
- Alpha-Beta pruning helps significantly **speed up the search process** by eliminating portions of the tree that can't possibly affect the final decision.

**1. Basics of Minimax:** In a Minimax search, you explore a game tree by considering both maximizing (player's) and minimizing (opponent's) moves at alternating levels. The goal is to find the best move while considering the worst-case scenario.

**2. Alpha and Beta Values:**
   - **Alpha (α):** Represents the best value that the maximizing player (the player whose turn it is) has found so far at any choice point along the path to the root.
   - **Beta (β):** Represents the best value that the minimizing player (the opponent) has found so far at any choice point along the path to the root.

**3. Pruning Process:**
   - When the algorithm explores a node at a maximizing level, it updates the alpha value with the maximum of the current alpha and the value of the node.
   - When exploring a node at a minimizing level, it updates the beta value with the minimum of the current beta and the value of the node.
   - Pruning occurs when a node is determined to be worse than the current alpha-beta bounds. If the value of a node at a maximizing level is less than or equal to alpha or the value of a node at a minimizing level is greater than or equal to beta, the search can stop at that node because it won't affect the final decision. Thus, the algorithm avoids further exploration of that branch.

**4. The Pruning Condition:**
   - For maximizing nodes: If the value of a child node is less than or equal to alpha, prune the subtree below this node.
   - For minimizing nodes: If the value of a child node is greater than or equal to beta, prune the subtree below this node.

**5. Benefits of Alpha-Beta Pruning:**
   - Alpha-Beta pruning significantly reduces the number of nodes that need to be evaluated in the search tree, especially in deep trees.
   - It prunes away branches that can't lead to a better solution, saving computation time.
   - In the best-case scenario, if the tree is perfectly balanced, Alpha-Beta pruning reduces the search effort to O(b^(d/2)), where "b" is the branching factor and "d" is the depth of the tree.

**Pseudocode for Alpha-Beta Pruning:**

Here's a simplified pseudocode representation of the Alpha-Beta pruning procedure:

```plaintext
function alphabeta(node, depth, alpha, beta, maximizingPlayer):
    if depth is 0 or node is a terminal node:
        return the heuristic value of the node
    
    if maximizingPlayer:
        for each child in node's children:
            alpha = max(alpha, alphabeta(child, depth - 1, alpha, beta, False))
            if beta <= alpha:
                break  # Beta cut-off
        return alpha
    else:
        for each child in node's children:
            beta = min(beta, alphabeta(child, depth - 1, alpha, beta, True))
            if beta <= alpha:
                break  # Alpha cut-off
        return beta
```

## Negamax algorithm

- A variation of the minimax algorithm. 
- The goal of Negamax is to find the **optimal move for the player taking the current turn**, assuming the **opponent will also make the best possible moves**. 
- It's commonly used in **combination with alpha-beta pruning** to improve the efficiency of the search.

**1. Minimax Principle:**
   - The fundamental concept behind Negamax is the minimax principle, which states that in a zero-sum game, each player aims to minimize their opponent's potential score while maximizing their own score.

**2. Recursive Search:**
   - Negamax uses a recursive depth-first search to explore possible game states, typically represented as a tree of game positions.

**3. Evaluation Function:**
   - An evaluation function is used to estimate the value of a game position. In chess, this function might assign values to pieces, board control, and other factors.

**4. Terminal Nodes:**
   - The search terminates when a terminal node is reached, representing the end of the game (win, lose, or draw). The terminal nodes are assigned values based on the outcome.

**5. Backward Propagation:**
   - As the search unwinds, values are propagated back up the tree. Since it's a zero-sum game, the value of a node at an even depth (the player's turn) is negated.

**6. Alpha-Beta Pruning:**
   - Alpha-beta pruning is often applied to reduce the number of nodes that need to be evaluated. It eliminates branches of the search tree that can't possibly affect the final decision.

**7. Transposition Tables:**
   - To avoid re-evaluating the same game positions, transposition tables (also known as hash tables) can be used to store previously evaluated positions.

**8. Iterative Deepening:**
   - To explore deeper into the game tree in a time-constrained environment, iterative deepening can be employed. The search starts with a shallow depth and incrementally deepens until time runs out.


```python
def negamax(board, depth, player):
    if depth == 0 or board.is_terminal():
        return player * evaluate(board)  # Evaluation function

    max_value = float('-inf')
    for move in board.get_legal_moves():
        board.make_move(move, player)
        value = -negamax(board, depth - 1, -player)
        board.undo_move(move)

        if value > max_value:
            max_value = value

    return max_value

def best_move(board, depth, player):
    legal_moves = board.get_legal_moves()
    best_score = float('-inf')
    best_move = None

    for move in legal_moves:
        board.make_move(move, player)
        score = -negamax(board, depth - 1, -player)
        board.undo_move(move)

        if score > best_score:
            best_score = score
            best_move = move

    return best_move
```

## [easyAI](https://github.com/Zulko/easyAI) library

## Last Coin Standing

- A popular game mechanics concept that can be applied to various games, both physical and digital. 
- The mechanics involve a competition where players take turns removing or interacting with items (often represented as coins) from a shared space or pool, with the objective of strategically positioning themselves to be the last player with a move. 
- It's a simple yet engaging concept that can be used in different games and variations. 

**Objective:**
- The primary objective of the game is to be the last player who can make a legal move, typically by taking or manipulating the coins or items in the game.

**Game Components:**
- **Coins or Tokens:** These are the items that players take turns removing or interacting with. The coins are usually placed in a central area, and players choose a specific number of coins to take during their turn.

**Rules:**
- **Turn-Based:** The game is played in turns, with each player taking one turn at a time. The order of play can be determined by random selection, such as a coin toss.

- **Move Restrictions:** Players can typically choose a specific number of coins to take on their turn. The exact rules for move restrictions can vary. For example, players may be required to take 1, 2, or 3 coins per turn.

- **Legal Moves:** Players must adhere to the specified move restrictions, and they can only take coins if there are enough remaining in the central pool to meet the move requirement. If there are not enough coins, they may have to take fewer or even no coins.

- **Strategic Choices:** The game involves a significant element of strategy. Players need to decide how many coins to take and when to take them. The objective is to force opponents into positions where they can't make a legal move.

- **Winning Condition:** The game continues until one player is left as the last one who can make a move. That player is declared the winner.

**Variations:**
- **Different Move Restrictions:** Variations can involve different restrictions on the number of coins that can be taken during a turn. For example, the move restriction can be 1, 2, 3, or any other number.

- **Multiple Pools:** Some versions of the game may include multiple pools of coins that players can choose from, adding complexity to the strategy.

- **Adding Special Rules:** Special rules or power-ups can be introduced to add variety and complexity to the game. For example, some coins may have special abilities or restrictions associated with them.

**Examples:**
- "NIM" is a classic mathematical game based on Last Coin Standing mechanics. In the standard game, players take turns removing coins from piles, with specific move restrictions.

## Tic-Tac-Toe


```python
from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax

class TicTacToe(TwoPlayersGame):
    def __init__(self, players):
        self.players = players
        self.board = [0] * 9  # 0 represents an empty space
        self.nplayer = 1  # Player 1 starts

    def possible_moves(self):
        return [i for i, s in enumerate(self.board) if s == 0]

    def make_move(self, move):
        self.board[move] = self.nplayer

    def unmake_move(self, move):
        self.board[move] = 0

    def lose(self):
        # Check for a losing condition
        for a, b, c in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return True
        return False

    def is_over(self):
        return self.lose() or (len(self.possible_moves()) == 0)

    def show(self):
        for i in range(0, 9, 3):
            print(self.board[i], self.board[i + 1], self.board[i + 2])
        print("\n")

if __name__ == "__main__":
    # Create the game with a human player and an AI player using the Negamax algorithm
    ai_algo = Negamax(6)
    game = TicTacToe([Human_Player(), AI_Player(ai_algo)])

    # Start the game
    game.play()
```

## Connect Four

```python
from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax
import numpy as np

class ConnectFour(TwoPlayersGame):
    def __init__(self, players):
        self.players = players
        self.board = np.zeros((6, 7), dtype=int)  # 0 represents an empty space
        self.nplayer = 1  # Player 1 starts

    def possible_moves(self):
        return [i for i in range(7) if self.board[0][i] == 0]

    def make_move(self, move):
        row = 0
        while row < 6 and self.board[row][move] == 0:
            row += 1
        self.board[row - 1][move] = self.nplayer

    def unmake_move(self, move):
        row = 0
        while row < 6 and self.board[row][move] == 0:
            row += 1
        self.board[row][move] = 0

    def lose(self):
        # Check for a losing condition
        for player in [1, 2]:
            for r in range(6):
                for c in range(7):
                    if self.board[r][c] == player:
                        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                            for dist in range(1, 4):
                                nr, nc = r + dr * dist, c + dc * dist
                                if nr < 0 or nr >= 6 or nc < 0 or nc >= 7:
                                    break
                                if self.board[nr][nc] != player:
                                    break
                                if dist == 3:
                                    return True
        return False

    def is_over(self):
        return self.lose() or len(self.possible_moves()) == 0

    def show(self):
        for row in self.board:
            print(" ".join(["X" if cell == 1 else "O" if cell == 2 else "." for cell in row]))
        print("1 2 3 4 5 6 7")

if __name__ == "__main__":
    # Create the game with a human player and an AI player using the Negamax algorithm
    ai_algo = Negamax(6)
    game = ConnectFour([Human_Player(), AI_Player(ai_algo)])

    # Start the game
    game.play()
```

## Hexapawn

```python
from easyAI import TwoPlayersGame, Human_Player, AI_Player, Negamax

class Hexapawn(TwoPlayersGame):
    def __init__(self, players):
        self.players = players
        self.board = ["  " for _ in range(9)]  # Initialize the board
        self.nplayer = 1  # Player 1 starts

    def possible_moves(self):
        moves = []
        for i, piece in enumerate(self.board):
            if piece == f"P{self.nplayer}":
                if i >= 3 and self.board[i - 3] == "  ":
                    moves.append((i, i - 3))
                if i >= 2 and i % 3 != 0 and self.board[i - 2] == f"P{3 - self.nplayer}":
                    moves.append((i, i - 2))
                if i >= 4 and i % 3 != 2 and self.board[i - 4] == f"P{3 - self.nplayer}":
                    moves.append((i, i - 4))
        return moves

    def make_move(self, move):
        from_index, to_index = move
        self.board[to_index] = self.board[from_index]
        self.board[from_index] = "  "

    def unmake_move(self, move):
        from_index, to_index = move
        self.board[from_index] = self.board[to_index]
        self.board[to_index] = "  "

    def lose(self):
        for i in range(6, 9):
            if self.board[i] == f"P{self.nopponent}":
                return True
        return False

    def is_over(self):
        return self.lose() or len(self.possible_moves()) == 0

    def show(self):
        for i in range(0, 9, 3):
            print(" | ".join(self.board[i:i + 3]))
            if i < 6:
                print("-" * 9)

if __name__ == "__main__":
    # Create the game with a human player and an AI player using the Negamax algorithm
    ai_algo = Negamax(7)
    game = Hexapawn([Human_Player(), AI_Player(ai_algo)])

    # Start the game
    game.play()
```

## Summary