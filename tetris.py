import numpy as np
import cv2
import utils.display as display
from mino import Mino

CNT = 100


class Tetris():
    def __init__(self, cfg):
        self.init(cfg)

    def init(self, cfg):

        self.board = self.init_board(cfg)
        self.minos = self.init_minos()
        self.score = self.init_score()
        self.rate = self.init_rate()
        self.chain = self.init_chain()

    def init_board(self, cfg):
        board = np.zeros((cfg.N_H + 6, cfg.N_W + 6))
        board[:, 2] = 1
        board[:, -3] = 1
        board[-3, :] = 1
        return board

    def init_minos(self):
        return [Mino(4, 0, np.random.randint(7)) for i in range(5)]

    def init_score(self):
        return 0

    def init_rate(self):
        return 1

    def init_chain(self):
        return 0

    def update_left(self, tmp_mino):
        tmp_mino.x -= 1

    def update_right(self, tmp_mino):
        tmp_mino.x += 1

    def update_leftRotate(self, tmp_mino):
        tmp_mino.degree = (tmp_mino.degree + 3) % 4
        tmp_mino.mino = np.array(tmp_mino.mino_list[tmp_mino.degree])

    def update_rightRotate(self, tmp_mino):
        tmp_mino.degree = (tmp_mino.degree + 1) % 4
        tmp_mino.mino = np.array(tmp_mino.mino_list[tmp_mino.degree])

    def update_down(self, tmp_mino):
        tmp_mino.y += 1
        self.update_collision(tmp_mino)

    def update_quick(self, tmp_mino):
        while not tmp_mino.collision(self.board):
            tmp_mino.y += 1
        self.update_collision(tmp_mino)

    def update_collision(self, mino):
        if mino.collision(self.board):
            self.minos[0].y = mino.y - 1
            self.update_board(self.minos[0])
            self.minos = self.minos[1:]
            self.minos.append(Mino(5, 0, np.random.randint(7)))

    def update_post(self, mino):
        if not mino.collision(self.board):
            self.minos[0].copyfrom(mino)

    def update_mino(self, key):
        tmp_mino = self.minos[0].copyto()
        if key == "leftRotate":  # 左回転
            self.update_leftRotate(tmp_mino)
        elif key == "rightRotate":  # 右回転
            self.update_rightRotate(tmp_mino)
        elif key == "right":  # 右移動
            self.update_right(tmp_mino)
        elif key == "left":  # 左移動
            self.update_left(tmp_mino)
        elif key == "down":
            self.update_down(tmp_mino)
        elif key == "quick":
            self.update_quick(tmp_mino)
        self.update_post(tmp_mino)

        return self.check_dead()

    def update_board(self, mino):
        h, w = mino.mino.shape
        self.board[mino.y:mino.y + h, mino.x:mino.x + w] += mino.mino
        self.check_line()

    def check_line(self):
        h, w = self.board.shape
        tmp = self.board[:h - 3, 3:w - 3]
        tmp = tmp[np.any(tmp == 0, axis=1)]
        a = (h - 3) - tmp.shape[0]

        if a > 0:
            self.chain += 1
            if self.chain >= 3:
                self.rate *= 1.1
            if a == 4:
                self.score += self.rate * 80
            else:
                self.score += self.rate * a * 10
        else:
            self.chain = self.init_chain()
            self.rate = self.init_rate()
        zero = np.zeros((a, 10))
        tmp = np.concatenate([zero, tmp])
        self.board[:h - 3, 3:w - 3] = tmp

    def check_dead(self):
        return np.any(self.board[3, 3:13] > 0)
