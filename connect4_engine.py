import sys
import torch
import torch.nn as nn
import random
import numpy as np
import math
import copy
import pickle
#from torch.multiprocessing import Pool, set_sharing_strategy, set_start_method

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
                #print(f"Player {player} wins")
                #print_board()
                reset_board()
                return WIN
            check_draw()
            if check_draw():
                #print('Draw!')
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

# invert the board, so that 1's become 2's and vice-versa
def invert_board(board):
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == 1:
                board[r][c] = 2
            elif board[r][c] == 2:
                board[r][c] = 1

# represents the data of an AI model
class Connect4Model(nn.Module):
  
    def __init__(self, input_dim, output_dim):
        super(Connect4Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding="same")
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(ROWS*COLS, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.win_count = 0
        # list of (board_state tensor, move) tuples
        self.moves = []

    def __lt__(self, other):
        return self.win_count < other.win_count

    # this function tells pytorch how to apply the layers of the neural net
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x).flatten()
        x = self.fc1(x)
        return self.fc2(x)

    # save the model to a file
    def save(self, filename):
        torch.save(self.state_dict(), filename)
    
    # load the model from a file
    def load(self, filename):
        self.load_state_dict(torch.load(filename, weights_only=True))
        self.eval()

    # ask the model which move it should play, given the board
    # position and this model's player.  Returns a column to play in.
    def play_move(self, board, player):
        # construct a mask of valid values - an entry in the list is
        # True if it's legal to move there
        mask = torch.tensor([0.0 if board[0, i] == 0 else math.inf for i in range(COLS)]).cuda()
        if player == 2:
            invert_board(board)
        board_tensor = torch.tensor(board, dtype=torch.float32).reshape(1, 1, ROWS, COLS).cuda()
        with torch.no_grad():
            outputs = self.forward(board_tensor)
        if player == 2:
            invert_board(board)
        # apply the mask to the outputs and return the highest-rated move
        output_tensor = outputs - mask
#        print("move = " + str(outputs))
        move = torch.argmax(output_tensor).item()
        self.moves.append((board_tensor.detach(), move))
        return move
    
    # return a copy of this model, with some weights randomly mutated
    def get_mutant(self):
        mutant = copy.deepcopy(self)
        # modify the mutant
        with torch.no_grad():
            for param in mutant.parameters():
                for val in param.flatten():
                    # 1% mutation rate
                    if random.random() < 0.01:
                        # mutate the parameter by multiplying by the range -2.0 to 2.0
                        val *= random.uniform(-2.0, 2.0)
        return mutant


def reward_move(model, board_tensor, move, reward, optimizer, criterion):
    optimizer.zero_grad()
    outputs = model(board_tensor)
    # we can make the target have the same outputs, but with a
    # slightly larger value for this move
    target = outputs.clone()
    if reward > 0:
        target[move] = torch.max(outputs) + reward
    else:
        target[move] = torch.min(outputs) + reward
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

def reward_model(model, reward):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # reward the last move
    reward_move(model, model.moves[-1][0], model.moves[-1][1], reward, optimizer, criterion)

    # remove the winning move and reward a random set of 5 of the
    # remaining moves
    if reward == 1.0:
        model.moves.pop()
        random.shuffle(model.moves)
        for move in model.moves[:5]:
            reward_move(model, move[0], move[1], reward, optimizer, criterion)

# have two models play a game until one wins.
# if a model is None then a human needs to play.
# needs to return which model wins the game.
# Also need a way to say it was a draw
def play_one_game(model1, model2):
    players = [model1, model2]
    reset_board()
    if model1 is not None:
        model1.moves.clear()
    if model2 is not None:
        model2.moves.clear()
    player_index = random.randint(0,1)
    while True:
        m = players[player_index]
        if m is None:
            print_board()
            print(f"Type a column number. Player {player}'s move:")

            try:
                move = int(input())
                if move == 0:
                    print('Quiting...')
                    sys.exit(0)
                move = move - 1

            except (ValueError, IndexError):
                print("Invalid move! Type a valid column, try again")
                continue
        else:
            move = m.play_move(board, player_index + 1)
            #print(f'AI player {player} moved in column {move + 1}')
        result = drop_piece(move)
        if result == DRAW:
            return
            break
        elif result == WIN:
            # reinforce winning move
            if m is not None:
                reward_model(m, 1.0)
            # punish losing move
            other_m = players[(player_index + 1) % 2]
            reward_model(other_m, -1.0)
            return player_index
            break
        elif result == INVALID_MOVE:
            if m is None:
                # human made a mistake, let them try again
                continue
            else:
                print("AI was not supposed to make an invalid move")
                sys.exit(0)

        player_index = (player_index + 1) % 2

# Play a series of games between two models, and return the win counts of each
def play_series(model1, model2, number_of_games):
    model1_wins = 0
    model2_wins = 0
    for i in range(number_of_games):
        result = play_one_game(model1, model2)
        if result == 0:
            model1_wins += 1
        elif result == 1:
            model2_wins += 1
    return (model1_wins, model2_wins)

# play the ith model against models numbered higher in the given list of models
def play_i(i, models, number_of_games):
    all_wins = [0] * len(models)
    for j in range(i+1, len(models)):
        wins = play_series(models[i], models[j], 100)
        print(f"model {i} won {wins[0]}, model {j} won {wins[1]}")
        all_wins[i] = wins[0]
        all_wins[j] = wins[1]
    return all_wins

class GeneticAlgorithm:
    def __init__(self, num_models):
        self.num_models = num_models
        self.models = []
        self.generation = 0
        for i in range(num_models):
            self.models.append(Connect4Model(ROWS * COLS, COLS))

    # updates win count on each model
    def everyone_play_everyone(self):
        # initialize everyone's count to 0
        for model in self.models:
            model.win_count = 0

        def store_result(all_wins):
            for m, w in zip(self.models, all_wins):
                m.win_count += w

#        # allocate a list to store async results from running models in parallel
#        async_results = [None] * self.num_models
#
#        # assume a 12-thread CPU for now
#        with Pool(12) as p:
#            # async play each model against the rest
        for i in range(self.num_models):
            store_result(play_i(i, self.models, 100))

    # mutate survivors until population is full
    def fill_population(self):
        survivor_count = len(self.models)
        while len(self.models) < self.num_models:
            to_mutate = self.models[random.randint(0, survivor_count - 1)]
            self.models.append(to_mutate.get_mutant())
 
    def generation_step(self):
        # have everyone play everyone else
        self.everyone_play_everyone()

        # rank the models by decreasing number of wins
        self.models.sort(reverse=True)

        print(f"generation {self.generation} best model won {self.models[0].win_count} games, worst won {self.models[-1].win_count}")

        # keep only the top half of the models
        num_keep = self.num_models // 2
        self.models = self.models[:num_keep]

        # mutate the surviving models into a new population
        self.fill_population()
        self.generation += 1


if __name__ == "__main__":
#    set_start_method('spawn')

#    model = Connect4Model(ROWS * COLS, COLS)
#    play_one_game(None, model)

    try:
        with open('genetic_data', 'rb') as f:
            gen = pickle.load(f)
            gen.num_models = 40
    except:
        gen = GeneticAlgorithm(40)
        for model in gen.models:
            torch.nn.init.uniform_(model.conv1.weight)
            torch.nn.init.uniform_(model.conv2.weight)
            torch.nn.init.uniform_(model.bn.weight)
            torch.nn.init.uniform_(model.fc1.weight)
            torch.nn.init.uniform_(model.fc2.weight)

    for model in gen.models:
        model.cuda()

    for _ in range(2):
        gen.generation_step()
        with open('genetic_data', 'wb') as f:
            pickle.dump(gen, f)
#    play_one_game(None, gen.models[1])
