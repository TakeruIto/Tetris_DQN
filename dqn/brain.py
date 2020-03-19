from dqn.model import get_model
import random
import copy
import numpy as np
from collections import namedtuple
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from dqn.memory import ReplayMemory


class Brain:
    def __init__(self, cfg,tetris):
        self.num_actions = cfg.MODEL.SIZE_ACTION
        self.gamma = cfg.SOLVER.GAMMA
        self.BATCH_SIZE = cfg.SOLVER.BATCH_SIZE

        transition = namedtuple(
            'Transicion', ('state', 'action', 'next_state', 'reward'))
        self.memory = ReplayMemory(cfg.SOLVER.CAPACITY, transition)
        self.model = get_model(cfg)

        self.target_net = copy.deepcopy(self.model)
        self.target_net.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.tetris = tetris
    def replay(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])

        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)

        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state)))

        next_state_values = torch.zeros(BATCH_SIZE)

        self.target_net.eval()
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_net.load_state_dict(self.model.state_dict())

    def decide_action(self, state,mino, episode):
        epsilon = 0.41 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.tetris.get_masked_action(self.model(state), mino)
        else:
            action = torch.LongTensor(
                [[self.tetris.get_random_masked_action(mino)]])

        return action

    def brain_predict(self, state):
        self.model.eval()
        with torch.no_grad():
            action = self.model(state).max(1)[1].view(1, 1)
        return action
