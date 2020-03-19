from dqn.brain import Brain


class Agent:
    def __init__(self, cfg, tetris):
        self.brain = Brain(cfg, tetris)

    def update_q_network(self):
        self.brain.replay()

    def update_target_model(self):
        self.brain.update_target_model()

    def get_action(self, state, mino, episode):
        action = self.brain.decide_action(state, mino, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def predict_action(self, state):
        action = self.brain.brain_predict(state)
        return action
