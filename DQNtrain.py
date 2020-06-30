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





Average_Q_lengths = []

params_dict = []  # for graph writing
sum_q_lens = 0
AVG_Q_len_perepisode = []

transition_time = 8




LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.005
GAMMA = 0.99  # discount factor of reward
TRAIN_EPISODE = 100  # 训练的总episode数

def run_episode(agent, env, rpm):
    traci.load(["--start", "-c", "data/cross.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])

    obs = getState_baseline(transition_time)
    steps = 0
    total_reward = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        steps += 1
        print("steps:",steps)
        action = agent.sample(obs)
        print("sample,",action)

        same_action_count = 0
        for temp in reversed(rpm.buffer):
            if temp[1] == 0:
                same_action_count += 1
            else:
                break
        if same_action_count == 20:
            action = 1
            print("SAME ACTION PENALTY")

        else:
            print("POLICY FOLLOWED ")

        print("action:", action)
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

        while True:
            action = agent.predict(obs)

            queueLength = getQueueLength()
            next_obs = makeMove(action, transition_time)
            new_queueLength = getQueueLength()
            reward = getReward(queueLength, new_queueLength)
            isOver = traci.simulation.getMinExpectedNumber()

            obs = next_obs
            episode_reward += reward
            if not isOver:
                break
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

        print("=============================")
        print("episode:",episode)
        total_reward, steps = run_episode(agent, env, rpm)
        episode += 1

        eval_reward = evaluate(env, agent, render=False)
        logger.info('episode:{}    test_reward:{}'.format(
            episode, eval_reward))

        save_path = './dqnmodel/model_{}_{}.ckpt'.format(episode, total_reward)
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
