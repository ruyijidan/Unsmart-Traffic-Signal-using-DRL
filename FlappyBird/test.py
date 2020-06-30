from ple.games.flappybird import FlappyBird
from ple import PLE

import numpy as np
import parl
from parl.utils import logger

from FlappyBird.flappybird_model import Model
from FlappyBird.flappybird_agent import Agent

from FlappyBird.replay_memory import ReplayMemory


import os

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1024  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 1024   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.0001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等


def run_episode(agent, penv, rpm):
    total_reward = 0
    penv.reset_game()
    obs = list(penv.getGameState().values())
    step = 0
    while True:
        step += 1
        #print(step)
        action_index = agent.sample(obs)
        action = penv.getActionSet()[action_index]


        #print(action_index,action)
        # 行动
        reward = penv.act(action)
        next_obs = list(penv.getGameState().values())
        #isOver=bool(isOver)
        obs = np.squeeze(obs)
        next_obs = np.squeeze(next_obs)
        isOver = penv.game_over()
        #print((obs, action, reward, next_obs, isOver))
        rpm.append((obs, action_index , reward, next_obs, isOver))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,batch_isOver) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,batch_next_obs, batch_isOver)

        total_reward += reward
        obs = next_obs
        if isOver:
            break
    return total_reward


def evaluate(agent, penv, render=False):
    # test part, run 5 episodes and average
    eval_reward = []
    for i in range(5):
        penv.reset_game()
        obs = list(penv.getGameState().values())
        episode_reward = 0
        isOver = False
        while not isOver:
            action_index = agent.predict(obs)  # 选取最优动作
            action = penv.getActionSet()[action_index]
            reward = penv.act(action)
            obs = list(penv.getGameState().values())
            if render:
                penv.getScreenRGB()
            isOver=penv.game_over()
            episode_reward += reward
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    env = FlappyBird()
    penv = PLE(env, fps=30, display_screen=True,force_fps=True)
    #penv.init()
    np.random.seed(0)

    obs_shape = len(penv.getGameState())
    IMG_shape = penv.getScreenGrayscale().shape
    action_dim = len(penv.getActionSet())


    print(obs_shape,action_dim)

    rpm = ReplayMemory(MEMORY_SIZE)

    model = Model(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(
        model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape,
        act_dim=action_dim,
        e_greed=0.15,  # explore  0.1
        e_greed_decrement=1e-6   #1e-6
    )  # probability of exploring is decreasing during training




    # 加载模型
    if os.path.exists('./dqn_model.ckpt'):
        save_path = './dqn_model.ckpt'
        agent.restore(save_path)
        print("模型加载成功")
    eval_reward = evaluate(agent, penv)

if __name__ == '__main__':

    main()