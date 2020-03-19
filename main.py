import argparse

from tetris import Tetris
from utils.display import Window
from config.config import get_cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def main():
    cfg = get_cfg()
    args = get_args()
    tetris = Tetris(cfg)
    window = Window(tetris, cfg, args)
    window.mainloop()
    # tetris.start()


if __name__ == '__main__':
    main()
