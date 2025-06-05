import argparse

from tetris import Tetris
from utils.display import Window
from config.config import get_cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--color-theme', type=str, default='DEFAULT', 
                        choices=['DEFAULT', 'PASTEL', 'DARK', 'NEON'],
                        help='Choose color theme for Tetris blocks')
    return parser.parse_args()


def main():
    args = get_args()
    cfg = get_cfg(args.color_theme)
    tetris = Tetris(cfg)
    window = Window(tetris, cfg, args)
    window.mainloop()
    # tetris.start()


if __name__ == '__main__':
    main()
