import numpy as np
import cv2
import tkinter as tk


class Controller():

    def __init__(self, master, tetris, view, cfg):
        self.master = master
        self.tetris = tetris
        self.view = view
        self.UPDATE_TIME = cfg.UPDATE_TIME

        self.bindid_left = self.master.bind("<Left>", self.leftRotate)
        self.bindid_right = self.master.bind("<Right>", self.rightRotate)
        self.bindid_key = self.master.bind("<Key>", self.move)

        self.loop = self.master.after(self.UPDATE_TIME, self.update)

    def move(self, event):
        key = event.char
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

        if flag:
            self.finish_loop()

    def leftRotate(self, event):
        flag = self.tetris.update_mino("leftRotate")
        self.view.draw()

        if flag:
            self.finish_loop()

    def rightRotate(self, event):
        flag = self.tetris.update_mino("rightRotate")
        self.view.draw()

        if flag:
            self.finish_loop()

    def update(self):  # ループ
        flag = self.tetris.update_mino("down")
        self.view.draw()

        self.loop = self.master.after(self.UPDATE_TIME, self.update)

        if flag:
            self.finish_loop()

    def finish_loop(self):
        self.master.after_cancel(self.loop)
        self.master.unbind("<Left>", self.bindid_left)
        self.master.unbind("<Right>", self.bindid_right)
        self.master.unbind("<Key>", self.bindid_key)


class View():

    def __init__(self, master, tetris, cfg):
        self.master = master
        self.tetris = tetris
        self.colors = cfg.COLORS
        self.h_box = cfg.H_BOX
        self.w_box = cfg.W_BOX
        self.n_h = cfg.N_H
        self.n_w = cfg.N_W

        self.canvas = tk.Canvas(self.master, width=self.w_box * (self.n_w + 2) +
                                100, height=self.h_box * (self.n_h + 5), bg="black")  # キャンバスの作成
        self.canvas.pack()

        self.draw()

    def draw(self):

        self.canvas.delete("block")

        self.draw_board()
        self.draw_mino(self.tetris.minos[0])
        self.draw_next(self.tetris.minos[1:])
        self.draw_score()

    def draw_board(self):
        for i in range(self.n_h + 4):  # ブロックを表示
            for j in range(self.n_w + 2):
                x = self.w_box * j
                y = self.h_box * i
                self.canvas.create_rectangle(
                    x, y, x + self.w_box, y + self.h_box, fill="#{:02x}{:02x}{:02x}".format(
                        *self.colors[int(self.tetris.board[i][j + 2])]), outline="white", tag="block")
        self.canvas.create_line(0, self.h_box * 4,
                                self.w_box * (self.n_w + 2), self.h_box * 4, fill='red')

    def draw_mino(self, mino):
        x, y = (mino.x - 2) * self.w_box, mino.y * self.h_box
        h, w = mino.mino.shape
        for tmpy in range(h):
            for tmpx in range(w):
                if mino.mino[tmpy][tmpx]:
                    c = self.colors[mino.type_ + 2]
                    self.canvas.create_rectangle(
                        x + tmpx * self.w_box, y + tmpy * self.h_box, x + (tmpx + 1) * self.w_box, y + (tmpy + 1) * self.h_box, fill="#{:02x}{:02x}{:02x}".format(*c), outline="white", tag="block")

    def draw_next(self, minos):
        sW, sH = 10, 10
        x, y = (self.n_w + 2) * self.w_box + 10, 10
        self.canvas.create_rectangle((x, y), (x + 80, y + 320), fill="white")
        for i, mino in enumerate(minos):
            h, w = mino.mino.shape
            offsetx = int(40 - sW * (w / 2))
            offsety = int(40 - sH * (h / 2))
            for tmpy in range(h):
                for tmpx in range(w):
                    if mino.mino[tmpy][tmpx]:
                        c = self.colors[mino.type_ + 2]
                        self.canvas.create_rectangle((x + offsetx + tmpx * sW, y + offsety + i * 80 + tmpy * sH),
                                                     (x + offsetx + (tmpx + 1) * sW, y +
                                                      offsety + i * 80 + (tmpy + 1) * sH),
                                                     fill="#{:02x}{:02x}{:02x}".format(*c))

    def draw_score(self):
        x, y = (self.n_w + 2) * self.w_box + 50, 380
        self.canvas.create_text(x, y, text='{}'.format(
            self.tetris.score), fill="white", tag="block")


class Application(tk.Frame):
    def __init__(self, master, tetris, cfg):
        super().__init__(master)
        x = cfg.W_BOX * (cfg.N_W + 2) + 100
        y = cfg.H_BOX * (cfg.N_H + 4)
        master.geometry("{}x{}".format(x, y))
        master.title("tetris")

        self.view = View(master, tetris, cfg)
        self.controller = Controller(master, tetris, self.view, cfg)


class Window():

    def __init__(self, tetris, cfg):
        win = tk.Tk()
        self.app = Application(master=win, tetris=tetris, cfg=cfg)

    def mainloop(self):
        self.app.mainloop()
