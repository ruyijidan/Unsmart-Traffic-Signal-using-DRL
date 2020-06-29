from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import datetime
from time import time
import matplotlib.pyplot as plt
import parl
from parl.utils import logger

from DQN_model import Model
from DQN_agent import Agent
from replay_memory import ReplayMemory

import os
import sys

from make_env import *

ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate

GAMMA = 0.99  # reward 的衰减因子
TAU = 0.001  # 软更新的系数
MEMORY_SIZE = int(1e4)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20  # 预存一部分经验之后再开始训练
BATCH_SIZE = 32
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 0.05  # 动作噪声方差

TRAIN_EPISODE = 1000  # 训练的总episode数

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
nA = 2

LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
GAMMA = 0.99  # discount factor of reward


def run_episode(agent, env, rpm):
    traci.load(["--start", "-c", "data/cross.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)

    obs = getState_baseline(transition_time)
    steps = 0
    total_reward = 0

    while traci.simulation.getMinExpectedNumber() > 0:

        steps += 1
        action = agent.sample(obs)

        queueLength = getQueueLength()
        next_obs = makeMove(action, transition_time)

        new_queueLength = getQueueLength()
        reward = getReward(queueLength, new_queueLength)
        isOver = traci.simulation.getMinExpectedNumber()

        rpm.append((obs, action, reward, next_obs, isOver))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (steps % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_isOver) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_isOver)

        total_reward += reward
        obs = next_obs

    return total_reward, steps


def evaluate(env, agent, render=False):
    # test part, run 5 episodes and average
    eval_reward = []
    for i in range(5):
        obs = getState_baseline(transition_time)
        episode_reward = 0
        isOver = False
        while not isOver:
            action = agent.predict(obs)

            queueLength = getQueueLength()
            next_obs = makeMove(action, transition_time)
            new_queueLength = getQueueLength()
            reward = getReward(queueLength, new_queueLength)
            isOver = traci.simulation.getMinExpectedNumber()

            obs = next_obs
            episode_reward += reward

        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    # 创建环境
    traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                 "--tripinfo-output", "tripinfo.xml"])

    traci.trafficlight.setPhase("0", 0)

    act_dim = 2
    obs_dim = 1440  # (10, 24, 6)

    # 使用PARL框架创建agent

    model = Model(act_dim)
    algorithm = parl.algorithms.DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(algorithm, obs_dim, act_dim)

    # 加载模型
    if os.path.exists('./DQNmodel.ckpt'):
        save_path = './DQNmodel.ckpt'
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

        save_path = './dqnmodel/dqn_model_{}_{}.ckpt'.format(i, total_reward)
        agent.save(save_path)

    # 保存模型到文件 ./model.ckpt
    agent.save('./DQNmodel.ckpt')


if __name__ == '__main__':
    main()

    # print(AVG_Q_len_perepisode)
    # import matplotlib.pyplot as plt
    #
    # plt.plot([x for x in range(num_episode)], [AVG_Q_len_perepisode], 'ro')
    # plt.axis([0, num_episode, 0, 10])
    # plt.show()
