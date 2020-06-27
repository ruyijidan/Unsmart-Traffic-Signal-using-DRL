from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import datetime
from time import time
import matplotlib.pyplot as plt
import parl
from parl.utils import logger

from paddle_DDPGmodel import Model
from paddle_DDPGagent import Agent
from replay_memory import ReplayMemory

import os
import sys
from parl.algorithms import DDPG
from make_env import *

ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate

GAMMA = 0.99  # reward 的衰减因子
TAU = 0.001  # 软更新的系数
MEMORY_SIZE = int(1e4)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20  # 预存一部分经验之后再开始训练
BATCH_SIZE = 128
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 1.05  # 动作噪声方差

TRAIN_EPISODE = 100000  # 训练的总episode数

num_episode = 16
discount_factor = 0.9
# epsilon = 1
epsilon_start = 1
epsilon_end = 0.4
epsilon_decay_steps = 3000

Average_Q_lengths = []

params_dict = []  # for graph writing
sum_q_lens = 0
AVG_Q_len_perepisode = []

transition_time = 8
target_update_time = 20

replay_memory_init_size = 350
replay_memory_size = 8000
batch_size = 32


def run_episode(agent, env, rpm):
    obs = getState_baseline(transition_time)

    counter = 0

    total_reward = 0
    steps = 0

    while True:
        steps += 1
        # batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(obs.astype('float32'))

        # 增加探索扰动, 输出限制在 [-1.0, 1.0] 范围内

        action = np.random.normal(action, NOISE)
        #print(action.shape)
        print("Inside episode counter", counter)

        if traci.simulation.getMinExpectedNumber() <= 0:
            traci.load(["--start", "-c", "data/cross.sumocfg",
                        "--tripinfo-output", "tripinfo.xml"])

        obs = getState_baseline(transition_time)

        queueLength = getQueueLength()
        next_obs = makeMove(action, transition_time)

        new_queueLength = getQueueLength()
        reward = getReward(queueLength, new_queueLength)
        done = traci.simulation.getMinExpectedNumber()
        # action = [action]  # 方便存入replaymemory
        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))

        if len(rpm) > MEMORY_WARMUP_SIZE:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):

        total_reward = 0
        steps = 0
        while True:
            obs = getState_baseline(transition_time)

            action = agent.predict(obs.astype('float32'))

            steps += 1

            queueLength = getQueueLength()
            next_obs = makeMove(action, transition_time)
            new_queueLength = getQueueLength()
            reward = getReward(queueLength, new_queueLength)
            done = traci.simulation.getMinExpectedNumber()

            obs = next_obs
            total_reward += reward

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


def main():
    # 创建环境

    traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])

    traci.trafficlight.setPhase("0", 0)
    np.random.seed(0)

    act_dim = 2
    obs_dim = 1440  # (10, 24, 6)

    # 使用PARL框架创建agent

    model = Model(act_dim)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)

    # 加载模型
    if os.path.exists('./model.ckpt'):
        save_path = './model.ckpt'
        agent.restore(save_path)
        print("模型加载成功")
    env = 0
    # 创建经验池
    rpm = ReplayMemory(MEMORY_SIZE)
    # 往经验池中预存数据
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(agent, env, rpm)

    episode = 0
    while episode < TRAIN_EPISODE:
        for i in range(50):
            total_reward, steps = run_episode(agent, env, rpm)
            episode += 1

        eval_reward = evaluate(env, agent, render=False)
        logger.info('episode:{}    test_reward:{}'.format(
            episode, eval_reward))

        save_path = './model/dqn_model_{}_{}.ckpt'.format(i, total_reward)
        agent.save(save_path)

    # 保存模型到文件 ./model.ckpt
    agent.save('./model.ckpt')


if __name__ == '__main__':
    main()

    print(AVG_Q_len_perepisode)
    # import matplotlib.pyplot as plt
    #
    # plt.plot([x for x in range(num_episode)], [AVG_Q_len_perepisode], 'ro')
    # plt.axis([0, num_episode, 0, 10])
    # plt.show()
