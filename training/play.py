import numpy as np
from random import randint, random


class board_game_2048():


    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int)
        self.number_of_action = 4
        self.game_over = False

        self.fill_cell(self.board)
        self.fill_cell(self.board)

    def reset(self):
        self.board = np.zeros((4, 4), dtype=np.int)
        self.game_over = False
        self.number_of_action = 4
        self.fill_cell(self.board)
        self.fill_cell(self.board)

    def move(self, direction):
        # 0: left, 1: up, 2: right, 3: down
        rotated_board = np.rot90(self.board, direction)
        cols = [rotated_board[i, :] for i in range(4)]
        new_board = np.array([self.move_left(col) for col in cols])
        return np.rot90(new_board, -direction)

    def move_left(self, col):
        new_col = np.zeros((4), dtype=col.dtype)
        j = 0
        previous = None
        for i in range(col.size):
            if col[i] != 0:  # number different from zero
                if previous is None:
                    previous = col[i]
                else:
                    if previous == col[i]:
                        new_col[j] = 2 * col[i]
                        j += 1
                        previous = None
                    else:
                        new_col[j] = previous
                        j += 1
                        previous = col[i]
        if previous is not None:
            new_col[j] = previous
        return new_col

    def fill_cell(self, board):
        i, j = (self.board == 0).nonzero()
        if i.size != 0:
            rnd = randint(0, i.size - 1)
            board[i[rnd], j[rnd]] = 2 * ((random() > .9) + 1)

    def apply_action(self, direction):
        new_board = self.move(direction)
        moved = False
        if (new_board == self.board).all():
            # move is invalid
            pass
        else:
            moved = True
            self.fill_cell(new_board)
        return (moved, new_board)

    def is_game_over(self):
        left_board, right_board, up_board, down_board = self.move(0), self.move(2), self.move(1), self.move(3)
        if (left_board == self.board).all() and (right_board == self.board).all() \
            and (up_board == self.board).all() and (down_board == self.board).all():
            return True

        return False


game = board_game_2048()

print(game.board)

"""
while True:
    direction = int(input('Enter action:'))
    moved, game.board = main_loop(game.board, direction)
    if moved is False:
        print("Impossible")
    else:
        print(game.board)
"""