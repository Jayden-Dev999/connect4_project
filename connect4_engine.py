import tkinter as tk
from tkinter import messagebox

ROWS, COLS = 6, 7
player = 1
board = [[0 for _ in range(COLS)] for _ in range(ROWS)]

def drop_piece(col):
    global player
    for row in reversed(range(ROWS)):
        if board[row][col] == 0:
            board[row][col] = player    
            if check_win(player):
                print(f"Player {player} wins")
                reset_board()
            player = 2 if player == 1 else 1
            return

def check_win(p):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c+i] == p for i in range(4)): return True
    for r in range(ROWS - 3):
        for c in range(COLS):
            if all(board[r+i][c] == p for i in range(4)): return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r+i][c+i] == p for i in range(4)): return True
            if all(board[r+3-i][c+i] == p for i in range(4)): return True
    return False

def reset_board():
    global board, player
    board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    player = 1

import sys

def print_board():
    for r in range(ROWS):
        for c in range(COLS):
            sys.stdout.write(f"{board[r][c]}")
        print("")
    print("")

if __name__ == "__main__":
    reset_board()
    print_board()
    drop_piece(0)
    drop_piece(5)
    print_board()