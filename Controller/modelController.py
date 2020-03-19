
import torch
import numpy as np
import time
from dqn.model import get_model


class Controller():

    def __init__(self, master, tetris, view, cfg):
        self.n_w = cfg.SYSTEM.N_W
        self.n_h = cfg.SYSTEM.N_H

        self.tetris = tetris
        self.view = view
        self.UPDATE_TIME = cfg.SYSTEM.UPDATE_TIME

        self.model = get_model(cfg).double()

    def loop(self):
        location = np.arange((self.n_w + 1) * 4)
        while True:
            flag = False
            brd, mino = self.tetris.get_state()
            state = torch.tensor(np.append(brd.flatten(), mino))
            with torch.no_grad():
                action = self.tetris.get_masked_action(self.model(state), mino)
                deg = action // (self.n_w + 1)
                loc = action % (self.n_w + 1)
                flag = self.update(deg, loc)
            if flag:
                break

    def move(self, key):
        flag = False
        if key == "a":
            flag = self.tetris.update_mino("left")
        elif key == "d":
            flag = self.tetris.update_mino("right")
        elif key == "s":
            flag = self.tetris.update_mino("down")
        elif key == "w":
            flag = self.tetris.update_mino("quick")
        self.view.draw()

    def leftRotate(self):
        flag = self.tetris.update_mino("leftRotate")
        self.view.draw()

    def rightRotate(self):
        flag = self.tetris.update_mino("rightRotate")
        self.view.draw()

    def update(self, degree, location):
        location = location + 1
        while self.tetris.minos[0].degree != degree:
            self.rightRotate()
            self.update_canvas_and_rest()
        x = self.tetris.minos[0].x
        if x > location:
            while self.tetris.minos[0].x > location:
                self.move("a")
                self.update_canvas_and_rest()
        elif x < location:
            while self.tetris.minos[0].x < location:
                self.move("d")
                self.update_canvas_and_rest()
        self.move("w")
        self.view.draw()

    def update_canvas_and_rest(self):
        time.sleep(0.5)
        self.view.master.update()
