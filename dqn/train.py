import numpy as np
import copy
import torch

from dqn.agent import Agent
from tetris import Tetris
class Train:

    def __init__(self, cfg):
        self.num_states = cfg.MODEL.SIZE_STATE
        self.num_actions = cfg.MODEL.SIZE_ACTION
        self.num_episodes = cfg.SOLVER.NUM_EPISODES

        self.tetris = Tetris(cfg)
        self.agent = Agent(cfg,self.tetris)

    def run(self):
        episode_10_list = np.zeros(10)
        episode_final = False
        reward_per_epoch = []
        lifetime_per_epoch = []

        for episode in range(self.num_episodes):

            self.tetris.init()
            brd, mino = self.tetris.get_state()
            observation = torch.tensor(np.append(brd.flatten(), mino))
            state = observation
            state = state.type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            # frames = [self.env.getScreenRGB()]

            cum_reward = 0
            t = 0
            step = 0

            if episode % 15 == 0:
                self.agent.update_target_model()

            while not self.tetris.check_dead():
                step += 1

                action = self.agent.get_action(state,mino, episode)
                self.tetris.update_state(action.squeeze())
                rew = self.tetris.score
                print(rew)
                t += 1
                brd, mino = self.tetris.get_state()
                observation_next = torch.tensor(np.append(brd.flatten(), mino))
                done = self.tetris.check_dead()

                # frames.append(self.env.getScreenRGB())

                # 報酬を与える。さらにepisodeの終了評価と、state_nextを設定する
                if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                    state_next = None  # 次の状態はないので、Noneを格納

                    # 直近10episodeの立てたstep数リストに追加
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))

                    # 罰則を与える
                    reward = torch.FloatTensor([-1.0])

                else:
                    if rew > 0:
                        reward = torch.FloatTensor([1.0])
                    else:
                        reward = torch.FloatTensor([0.0])

                    state_next = observation_next.type(torch.FloatTensor)
                    # state_next = torch.from_numpy(state_next).type(
                    #     torch.FloatTensor)
                    state_next = torch.unsqueeze(
                        state_next, 0)

                cum_reward += rew

                self.agent.memorize(state, action, state_next, reward)

                self.agent.update_q_network()

                state = state_next

                # 終了時の処理
                if done:
                    print('%d Episode: Finished after %d steps：10試行の平均step数 = %.1lf' % (
                        episode, step + 1, episode_10_list.mean()))
                    reward_per_epoch.append(cum_reward)
                    lifetime_per_epoch.append(step + 1)
                    break

            if episode_final is True:
                # 動画の保存と描画
                display_frames_as_gif(frames)
                break

            # 50エピソード毎にlogを出力
            if episode % PRINT_EVERY_EPISODE == 0:
                print("Episode %d finished after %f time steps" % (episode, t))
                print("cumulated reward: %f" % cum_reward)

            # 100エピソード毎にアニメーションを作成
            if episode % SHOW_GIF_EVERY_EPISODE == 0:
                print("len frames:", len(frames))
                display_frames_as_gif(frames)
                continue

            # 2000タイムステップ以上続いたアニメーションを作成
            if step > 2000:
                print("len frames:", len(frames))
                display_frames_as_gif(frames)



    # モデルの保存

    def save_model():
        torch.save(agent.brain.model.state_dict(), 'weight.pth')
