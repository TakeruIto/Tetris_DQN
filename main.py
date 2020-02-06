from tetris import Tetris
from utils.display import Window
from config import get_cfg

def main():
    cfg = get_cfg()
    window = Window(cfg)
    window.mainloop()
    # tetris = Tetris()
    # tetris.start()


if __name__ == '__main__':
    main()
