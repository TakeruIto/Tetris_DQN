import copy
cfg = dict(
    H_BOX=20,
    W_BOX=20,
    N_W=10,
    N_H=22,
    COLORS = [[0, 0, 0],
              [128, 128, 128],
              [0, 191, 255],
              [65, 105, 225],
              [255, 165, 0],
              [0, 255, 0],
              [220, 20, 60],
              [186, 85, 211],
              [255, 255, 0],
              ],
    UPDATE_TIME = 500,

)

def get_cfg():
    d = dict2(copy.deepcopy(cfg))
    return d

class dict2(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
