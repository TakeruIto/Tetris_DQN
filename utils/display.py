import numpy as np
import cv2
import tkinter as tk

W = 20
H = 20
COLORS = [[0, 0, 0],
          [128, 128, 128],
          [0, 191, 255],
          [65, 105, 225],
          [255, 165, 0],
          [0, 255, 0],
          [220, 20, 60],
          [186, 85, 211],
          [255, 255, 0],
          ]


def draw(board, minos, score):
    h, w = board.shape
    canvas = np.zeros((H * h, W * w + 100, 3))
    for y in range(h):
        for x in range(w):
            cv2.rectangle(canvas, (x * W, y * H), ((x + 1) * W, (y + 1) * H),
                          COLORS[int(board[y][x])], thickness=-1)
            cv2.rectangle(canvas, (x * W, y * H), ((x + 1) * W, (y + 1) * H),
                          (100, 100, 100), thickness=1)

    _draw_mino(canvas, minos[0])
    _draw_next(canvas, minos[1:], (W * w + 10, 10))
    _draw_score(canvas, score, (W * w + 10, 360))
    cv2.line(canvas, (0, H * 2), (W * w, H * 2),
             color=(255, 0, 0), thickness=2)
    canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imshow("tetris", canvas)


def _draw_mino(canvas, mino):
    x, y = mino.x - 2, mino.y
    h, w = mino.mino.shape
    for tmpy in range(h):
        for tmpx in range(w):
            if mino.mino[tmpy][tmpx]:
                c = COLORS[mino.type_ + 2]
                cv2.rectangle(canvas, ((x + tmpx) * W, (y + tmpy) * H), ((x + tmpx + 1) * W, (y + tmpy + 1) * H),
                              color=c, thickness=-1)
                cv2.rectangle(canvas, ((x + tmpx) * W, (y + tmpy) * H), ((x + tmpx + 1) * W, (y + tmpy + 1) * H),
                              (100, 100, 100), thickness=1)


def _draw_next(canvas, minos, top_left_xy):
    sW, sH = 10, 10
    x, y = top_left_xy
    cv2.rectangle(canvas, (x, y), (x + 80, y + 320),
                  (255, 255, 255), thickness=1)
    for i, mino in enumerate(minos):
        h, w = mino.mino.shape
        offsetx = int(40 - sW * (w / 2))
        offsety = int(40 - sH * (h / 2))
        for tmpy in range(h):
            for tmpx in range(w):
                if mino.mino[tmpy][tmpx]:
                    c = COLORS[mino.type_ + 2]
                    cv2.rectangle(canvas, (x + offsetx + tmpx * sW, y + offsety + i * 80 + tmpy * sH),
                                  (x + offsetx + (tmpx + 1) * sW, y +
                                   offsety + i * 80 + (tmpy + 1) * sH),
                                  color=c, thickness=-1)


def _draw_score(canvas, score, p):
    cv2.putText(canvas, str(score), p, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), thickness=2)


class Controller():

    UPDATE = 750

    def __init__(self, master, tetris):
        self.master = master
        self.tetris = tetris

        # self.master.after(self.UPDATE, self.update)

        self.master.bind("<Left>", self.leftRotate)
        self.master.bind("<Right>", self.rightRotate)
        self.master.bind("<Key>", self.move)

    def move(self, event):
        key = event.char
        if key == " ":
            pass
        elif key == "a":
            pass
        elif key == "d":
            pass
        elif key == "s":
            pass
        elif key == "w":
            pass

    def leftRotate(self, event):
        print("left")

    def rightRotate(self, event):
        print("right")

    def update(self):  # ループ

        # self.view.update()

        self.master.after(self.UPDATE, self.update)


class View():

    def __init__(self, master, tetris, cfg):
        self.master = master
        self.tetris = tetris
        self.colors = cfg.COLORS
        self.h_box = cfg.H_BOX
        self.w_box = cfg.W_BOX
        self.n_h = cfg.N_H
        self.n_w = cfg.N_W

        self.canvas = tk.Canvas(self.master, width=self.w_box * (self.n_w+2) +
                                100, height=self.h_box * (self.n_h+3), bg="black")  # キャンバスの作成
        self.canvas.pack()

        self.draw()

    def update(self):

        for i in range(self.n_h):  # ブロックを表示
            for j in range(self.n_w):
                x = self.w_box * j
                y = self.h_box * i
                if self.model.data[i][j] == 1 or self.model.data[i][j] == 2:  # dataが1か2のものを表示
                    self.canvas.create_rectangle(
                        x, y, x + 30, y + 30, fill="red", outline="white", tag="block")

    def draw(self):
        self.draw_board()
        self.draw_mino(self.tetris.minos[0])
        self.draw_next(self.tetris.minos[1:])

    def draw_board(self):
        for i in range(self.n_h+3):  # ブロックを表示
            for j in range(self.n_w+2):
                x = self.w_box * j
                y = self.h_box * i
                self.canvas.create_rectangle(
                    x, y, x + self.w_box, y + self.h_box, fill="#{:02x}{:02x}{:02x}".format(
                        *self.colors[int(self.tetris.board[i][j])]), outline="white", tag="block")
        self.canvas.create_line(0, self.h_box*2,
                                self.w_box*(self.n_w+2), self.h_box*2, fill='red')

    def draw_mino(self, mino):
        x, y = (mino.x - 2)*self.w_box, mino.y*self.h_box
        h, w = mino.mino.shape
        for tmpy in range(h):
            for tmpx in range(w):
                if mino.mino[tmpy][tmpx]:
                    c = self.colors[mino.type_ + 2]
                    self.canvas.create_rectangle(
                        x+tmpx*self.w_box, y+tmpy*self.h_box, x+(tmpx+1)*self.w_box, y+(tmpy+1)*self.h_box, fill="#{:02x}{:02x}{:02x}".format(*c), outline="white", tag="block")

    def draw_next(self, minos):
        sW, sH = 10, 10
        x, y = (self.n_w+2)*self.w_box+10, 10
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


class Application(tk.Frame):
    def __init__(self, master, tetris, cfg):
        super().__init__(master)
        x = cfg.W_BOX * (cfg.N_W+2) + 100
        y = cfg.H_BOX * (cfg.N_H+3)
        master.geometry("{}x{}".format(x, y))  # ウィンドウサイズ
        master.title("tetris")  # タイトル名

        # self.model = Model() #インスタンスmodelを生成
        self.controller = Controller(master, tetris)  # インスタンスcontrollerを生成
        self.view = View(master, tetris, cfg)  # インスタンスviewを生成
        #
        # self.controller.view = self.view #引数の追加


class Window():

    def __init__(self, tetris, cfg):
        win = tk.Tk()
        self.app = Application(master=win, tetris=tetris, cfg=cfg)

    def mainloop(self):
        self.app.mainloop()
