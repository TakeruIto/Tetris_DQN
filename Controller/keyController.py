

class Controller():

    def __init__(self, master, tetris, view, cfg):
        self.master = master
        self.tetris = tetris
        self.view = view
        self.UPDATE_TIME = cfg.SYSTEM.UPDATE_TIME

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
