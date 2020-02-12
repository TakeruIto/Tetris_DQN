import tkinter as tk
from Controller.keyController import Controller as keyController
from Controller.modelController import Controller as modelController


class View():

    def __init__(self, master, tetris, cfg):
        self.master = master
        self.tetris = tetris
        self.colors = cfg.SYSTEM.COLORS
        self.h_box = cfg.SYSTEM.H_BOX
        self.w_box = cfg.SYSTEM.W_BOX
        self.n_h = cfg.SYSTEM.N_H
        self.n_w = cfg.SYSTEM.N_W

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
        for i in range(self.n_h + 5):  # ブロックを表示
            for j in range(self.n_w + 2):
                x = self.w_box * j
                y = self.h_box * i
                self.canvas.create_rectangle(
                    x, y, x + self.w_box, y + self.h_box, fill="#{:02x}{:02x}{:02x}".format(
                        *self.colors[int(self.tetris.board[i][j + 2])]), outline="white", tag="block")
        self.canvas.create_line(0, self.h_box * 4,
                                self.w_box * (self.n_w + 2), self.h_box * 4, fill='red', tag="block")

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
        self.canvas.create_rectangle(
            (x, y), (x + 80, y + 320), fill="white", tag="block")
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
                                                     fill="#{:02x}{:02x}{:02x}".format(*c), tag="block")

    def draw_score(self):
        x, y = (self.n_w + 2) * self.w_box + 50, 380
        self.canvas.create_text(x, y, text='{}'.format(
            self.tetris.score), fill="white", tag="block")


class Application(tk.Frame):
    def __init__(self, master, tetris, cfg, args):
        super().__init__(master)
        x = cfg.SYSTEM.W_BOX * (cfg.SYSTEM.N_W + 2) + 100
        y = cfg.SYSTEM.H_BOX * (cfg.SYSTEM.N_H + 5)
        master.geometry("{}x{}".format(x, y))
        master.title("tetris")

        self.view = View(master, tetris, cfg)
        if args.test:
            self.controller = modelController(master, tetris, self.view, cfg)
            self.controller.loop()
        else:
            self.controller = keyController(master, tetris, self.view, cfg)


class Window():

    def __init__(self, tetris, cfg, args):
        win = tk.Tk()
        self.app = Application(master=win, tetris=tetris, cfg=cfg, args=args)

    def mainloop(self):
        self.app.mainloop()
