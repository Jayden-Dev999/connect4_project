import sys
import torch
import torch.nn as nn
import random
import numpy as np

# Connect 4 constants
ROWS, COLS = 6, 7
DRAW = 0
WIN = 1
KEEP_PLAYING = 2
INVALID_MOVE = 3

# Global board + player
player = 1
board = np.zeros((ROWS, COLS), dtype=int)

def print_board():
    for r in range(ROWS):
        print("".join(str(board[r][c]) for c in range(COLS)))
    print()

def reset_board():
    global board, player
    board = np.zeros((ROWS, COLS), dtype=int)
    player = 1

def check_win(p):
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r, c+i] == p for i in range(4)):
                return True
    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r+i, c] == p for i in range(4)):
                return True
    # Diagonal /
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i, c+i] == p for i in range(4)):
                return True
    # Diagonal \
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+3-i, c+i] == p for i in range(4)):
                return True
    return False

def check_draw():
    return all(board[0, c] != 0 for c in range(COLS))

def drop_piece(col):
    global player
    if col is None or col < 0 or col >= COLS:
        return INVALID_MOVE

    for row in reversed(range(ROWS)):
        if board[row, col] == 0:
            board[row, col] = player
            if check_win(player):
                result = WIN
            elif check_draw():
                result = DRAW
            else:
                result = KEEP_PLAYING

            # swap players
            if result == KEEP_PLAYING:
                player = 2 if player == 1 else 1

            return result

    return INVALID_MOVE


# --------------------------
#  Neural Network AI Model
# --------------------------

class Connect4Model(nn.Module):
    def __init__(self, input_dim=42, output_dim=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def play_move(self, board, player):
        """Choose highest scoring *valid* column."""
        board_tensor = torch.tensor(board, dtype=torch.float32).flatten()
        with torch.no_grad():
            scores = self.forward(board_tensor).tolist()

        ordered = sorted(range(7), key=lambda c: scores[c], reverse=True)

        for col in ordered:
            if board[0][col] == 0:
                return col

        return None

    def evolve(self):
        """Random mutation (very simple)."""
        with torch.no_grad():
            for p in self.parameters():
                p += 0.02 * torch.randn_like(p)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


# --------------------------
#  AI vs AI Training System
# --------------------------

def play_one_game(model1, model2, ai_side=1):

    reset_board()

    if ai_side == 1:
        players = [model1, model2]
    else:
        players = [model2, model1]

    player_index = 0

    while True:
        current_player_num = player_index + 1
        move = players[player_index].play_move(board, current_player_num)

        result = drop_piece(move)

        if result == DRAW:
            return DRAW
        if result == WIN:
            return current_player_num
        if result == INVALID_MOVE:
            return -1

        player_index = (player_index + 1) % 2


def play_series(model1, model2, games=100):
    stats = {1: 0, 2: 0, DRAW: 0, -1: 0}

    for i in range(games):
        ai_side = 1 if i % 2 == 0 else 2
        result = play_one_game(model1, model2, ai_side)
        stats[result] += 1

    return stats


# --------------------------
#  Training Example
# --------------------------

if __name__ == "__main__":
    print("Starting...")
    import time
    time.sleep(1)   # tiny delay to see that it is running

    modelA = Connect4Model()
    modelB = Connect4Model()

    for gen in range(20):
        stats = play_series(modelA, modelB, games=200)

        # Winner evolves less, loser evolves more
        if stats[1] > stats[2]:
            modelB.evolve()
        else:
            modelA.evolve()

    time.sleep(0.2)
    print("done")