from tetris import Tetris
from utils.display import Window
from config import get_cfg


def main():
    cfg = get_cfg()
    tetris = Tetris(cfg)
    window = Window(tetris, cfg)
    window.mainloop()
    # tetris.start()


if __name__ == '__main__':
    main()
