import tkinter as tk
import numpy as np
import pickle
import random
import copy
import os

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 0: empty, 1: X (human), -1: O (AI)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    def is_winner(self, player):
        # Check rows, columns, and diagonals
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def is_draw(self):
        return 0 not in self.board

    def make_move(self, row, col):
        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player
            self.current_player *= -1  # Switch player
            return True
        return False

def check_winner(board, player):
    # Helper function for heuristic calculation
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

def calculate_heuristic(board, chromosome):
    ai = -1  # AI is player -1
    human = 1

    # Feature 1: Immediate win opportunities for AI
    win_opportunity = 0
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                temp_board = board.copy()
                temp_board[row][col] = ai
                if check_winner(temp_board, ai):
                    win_opportunity += 1

    # Feature 2: Block human's immediate win opportunities
    block_opportunity = 0
    for row in range(3):
        for col in range(3):
            if board[row][col] == 0:
                temp_board = board.copy()
                temp_board[row][col] = human
                if check_winner(temp_board, human):
                    block_opportunity += 1

    # Feature 3: Center control
    center = 1 if board[1][1] == ai else 0

    # Feature 4: Corner control
    corners = sum([board[0][0] == ai, board[0][2] == ai, board[2][0] == ai, board[2][2] == ai])

    features = np.array([win_opportunity, block_opportunity, center, corners])
    return np.dot(features, chromosome)

def create_population(pop_size):
    return np.random.uniform(-1, 1, (pop_size, 4))

def evaluate_fitness(chromosome, games=50):
    fitness = 0
    for _ in range(games):
        game = TicTacToe()
        while True:
            if game.is_winner(1) or game.is_winner(-1) or game.is_draw():
                break
            if game.current_player == -1:  # AI's turn
                best_score = -np.inf
                best_move = None
                for row in range(3):
                    for col in range(3):
                        if game.board[row][col] == 0:
                            temp_game = copy.deepcopy(game)
                            temp_game.make_move(row, col)
                            score = calculate_heuristic(temp_game.board, chromosome)
                            if score > best_score:
                                best_score = score
                                best_move = (row, col)
                game.make_move(*best_move)
            else:  # Baseline opponent (random)
                empty = [(r, c) for r in range(3) for c in range(3) if game.board[r][c] == 0]
                if empty:
                    r, c = random.choice(empty)
                    game.make_move(r, c)
        if game.is_winner(-1):
            fitness += 1
    return fitness

def evolve(population, elite_size=10, mutation_rate=0.1):
    fitness = [(i, evaluate_fitness(c)) for i, c in enumerate(population)]
    sorted_indices = sorted(fitness, key=lambda x: x[1], reverse=True)
    elites = [population[i] for i, _ in sorted_indices[:elite_size]]
    
    new_pop = elites.copy()
    while len(new_pop) < len(population):
        p1, p2 = random.choices(elites, k=2)
        child = (p1 + p2) / 2  # Average crossover
        child += np.random.normal(0, mutation_rate, child.shape)
        new_pop.append(child)
    return np.array(new_pop)

def train_ai(generations=50, pop_size=100):
    print("Training AI... (This may take a few minutes)")
    population = create_population(pop_size)
    for gen in range(generations):
        population = evolve(population)
        best = population[0]
        print(f"Generation {gen+1}: Best Fitness = {evaluate_fitness(best)}")
    with open("best_ai.pkl", "wb") as f:
        pickle.dump(best, f)
    return best

class GameGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Unbeatable Tic-Tac-Toe AI (Genetic Algorithm)")
        self.game = TicTacToe()
        self.ai_weights = self.load_ai()
        self.buttons = [[None]*3 for _ in range(3)]
        for row in range(3):
            for col in range(3):
                self.buttons[row][col] = tk.Button(
                    self.root, text="", font=("Arial", 20), width=4, height=2,
                    command=lambda r=row, c=col: self.on_click(r, c)
                )
                self.buttons[row][col].grid(row=row, column=col, padx=5, pady=5)
        self.status = tk.Label(self.root, text="Your Turn (X)", font=("Arial", 14))
        self.status.grid(row=3, columnspan=3)
        self.reset_btn = tk.Button(self.root, text="New Game", command=self.reset)
        self.reset_btn.grid(row=4, columnspan=3)
        self.update_ui()

    def load_ai(self):
        if not os.path.exists("best_ai.pkl"):
            train_ai()
        with open("best_ai.pkl", "rb") as f:
            return pickle.load(f)

    def on_click(self, row, col):
        if self.game.make_move(row, col):
            self.update_ui()
            if not self.check_game_over():
                self.ai_move()
                self.update_ui()
                self.check_game_over()

    def ai_move(self):
        best_score = -np.inf
        best_move = None
        for row in range(3):
            for col in range(3):
                if self.game.board[row][col] == 0:
                    temp_game = copy.deepcopy(self.game)
                    temp_game.make_move(row, col)
                    score = calculate_heuristic(temp_game.board, self.ai_weights)
                    if score > best_score:
                        best_score = score
                        best_move = (row, col)
        if best_move:
            self.game.make_move(*best_move)

    def update_ui(self):
        for row in range(3):
            for col in range(3):
                cell = self.game.board[row][col]
                text = "X" if cell == 1 else "O" if cell == -1 else ""
                self.buttons[row][col].config(text=text)

    def check_game_over(self):
        if self.game.is_winner(1):
            self.status.config(text="You Win!", fg="green")
            return True
        elif self.game.is_winner(-1):
            self.status.config(text="AI Wins!", fg="red")
            return True
        elif self.game.is_draw():
            self.status.config(text="Draw!", fg="blue")
            return True
        return False

    def reset(self):
        self.game.reset()
        self.update_ui()
        self.status.config(text="Your Turn (X)", fg="black")

if __name__ == "__main__":
    GameGUI().root.mainloop()