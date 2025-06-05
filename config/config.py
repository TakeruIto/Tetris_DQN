import yaml
from attrdict import AttrDict

def get_cfg(color_theme='DEFAULT'):
    with open('config/config.yaml') as file:
        config = AttrDict(yaml.safe_load(file))
    
    with open('config/block_colors.yaml') as file:
        colors_config = AttrDict(yaml.safe_load(file))
    
    if color_theme in colors_config.BLOCK_COLORS:
        theme_colors = colors_config.BLOCK_COLORS[color_theme]
        config.SYSTEM.COLORS = [
            theme_colors.EMPTY,
            theme_colors.WALL,
            theme_colors.I_PIECE,
            theme_colors.J_PIECE,
            theme_colors.L_PIECE,
            theme_colors.S_PIECE,
            theme_colors.Z_PIECE,
            theme_colors.T_PIECE,
            theme_colors.O_PIECE
        ]
    
    return config
