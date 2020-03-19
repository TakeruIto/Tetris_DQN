import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_model(cfg):
    size_state = cfg.MODEL.SIZE_STATE
    size_hidden = cfg.MODEL.SIZE_HIDDEN
    size_action = cfg.MODEL.SIZE_ACTION
    model = Model(size_state, size_hidden, size_action)
    model_weights = cfg.MODEL.WEIGHTS
    if os.path.exists(model_weights):
        print("load weights from {}".format(model_weights))
        param = torch.load('weight.pth')
        model.load_state_dict(param)

    return model


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = F.relu(self.fc4(h))
        return y
