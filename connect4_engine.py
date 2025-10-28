import sys
import torch
import torch.nn as nn
import random
import numpy as np

ROWS, COLS = 6, 7
player = 1
board = np.zeros((ROWS,COLS), dtype=int)

def print_board():
    for r in range(ROWS):
        for c in range(COLS):
            sys.stdout.write(f"{board[r][c]}")
        print("")
    print("")

DRAW = 0
WIN = 1
KEEP_PLAYING = 2
INVALID_MOVE = 3

def drop_piece(col):
    global player
    for row in reversed(range(ROWS)):
        if board[row,col] == 0:
            board[row,col] = player  
            if check_win(player):
                print(f"Player {player} wins")
                print_board()
                reset_board()
                return WIN
            check_draw()
            if check_draw():
                print('Draw!')
                reset_board()
                return DRAW
            player = 2 if player == 1 else 1
            return KEEP_PLAYING
    print("Invalid move! You went above the board, try again")
    return INVALID_MOVE

def check_win(p):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r,c+i] == p for i in range(4)): return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r+i,c] == p for i in range(4)): return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i,c+i] == p for i in range(4)): return True
            if all(board[r+3-i,c+i] == p for i in range(4)): return True
    return False

def check_draw():
    for c in range(COLS):
        if board[0,c] == 0:
            return False
    return True

def reset_board():
    global board, player
    board = np.zeros((ROWS, COLS), dtype=int)
    player = 1

# represents the data of an AI model
class Connect4Model(nn.Module):
  
    def __init__(self, input_dim, output_dim):
        super(Connect4Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
      
    # save the model to a file
    def save(self, filename):
        pass
    
    # load the model from a file
    def load(self, filename):
        pass
    
    # ask the model which move it should play, given the board
    # position and this model's player.  Returns a column to play in.
    def play_move(self, board, player):
        pass
    
    # update the model somehow to try to make it stronger
    # maybe this is random updates?
    def evolve(self):
        pass

# have two models play a game until one wins.
# needs to return which model wins the game.
# Also need a way to say it was a draw
def play_one_game(model1, model2):
    players = [model1, model2]
    reset_board()
    player_index = random.randint(0,1)
    while True:
        move = players[player_index].play_move(board, player_index + 1)
        result = drop_piece(move)
        if result == DRAW:
            return
            break
        elif result == WIN:
            # reward this model, penalize other one?
            return player_index
            break
        elif result == INVALID_MOVE:
            # penalize this model
            return player_index + 2
            break
        player_index = (player_index + 1) % 2

# Play a series of games between two models, return
# which model won more.  Need to define how many more
# is enough for us to conclude that one model is stronger
# than the other.
def play_series(model1, model2, number_of_games):
    for i in range(1000):
      play_one_game()

if __name__ == "__main__":
    reset_board()
    while True:
        print_board()
        print(f"Type a column number. Player {player}'s move:")
        try:
            move = int(input())
            if move == 0:
                print('Quiting...')
                sys.exit(0)
            drop_piece(move - 1)

        except (ValueError, IndexError):
            print("Invalid move! Type a valid column, try again")
            continue

