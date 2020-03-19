from dqn.train import Train
from config.config import get_cfg

if __name__ == '__main__':

    cfg = get_cfg()

    dqn = Train(cfg)
    dqn.run()
    torch.save(flappy_env.agent.brain.model.state_dict(), 'weight.pth')
