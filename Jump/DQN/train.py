import gym
import numpy as np
import parl
from parl.utils import logger

from jump_model import Model
from jump_agent import Agent

from replay_memory import ReplayMemory

from jump_env import JumpGame
import os

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 512  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 256   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等


def run_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()
    obs = np.reshape(obs, (1, 13))

    step = 0
    while True:
        step += 1
        #print(step)
        action = agent.sample(obs)
        reward, next_obs,  isOver = env.step(action)
        #isOver=bool(isOver)
        obs = np.squeeze(obs)
        next_obs = np.squeeze(next_obs)
        # print(obs)
        # print("==========================================================")
        # print(next_obs)
        #print((obs, action, reward, next_obs, isOver))
        rpm.append((obs, action, reward, next_obs, isOver))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,batch_isOver) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,batch_next_obs, batch_isOver)

        total_reward += reward
        obs = next_obs
        if isOver:
            break
    return total_reward


def evaluate(agent, env, render=False):
    # test part, run 5 episodes and average
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        isOver = False
        while not isOver:
            action = agent.predict(obs)
            if render:
                env.render()
            reward, obs, isOver = env.step(action)
            #obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    env = JumpGame()
    np.random.seed(0)


    action_dim = 2
    obs_shape = 13



    rpm = ReplayMemory(MEMORY_SIZE)

    model = Model(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(
        model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape,
        act_dim=action_dim,
        e_greed=0.01,  # explore
        e_greed_decrement=1e-6
    )  # probability of exploring is decreasing during training




    # 加载模型
    if os.path.exists('./dqn_model.ckpt'):
        save_path = './dqn_model.ckpt'
        agent.restore(save_path)
        print("模型加载成功")

    while len(rpm) < MEMORY_WARMUP_SIZE:  # warm up replay memory
        run_episode(agent, env, rpm)

    max_episode = 500

    # start train
    episode = 0
    while episode < max_episode:

        # train part
        for i in range(0, 50):
            total_reward = run_episode(agent, env, rpm)
            episode += 1

        eval_reward = evaluate(agent, env)
        logger.info('episode:{}    test_reward:{}'.format(
            episode, eval_reward))
        save_path = './model/dqn_model_{}_{}.ckpt'.format(episode,eval_reward)
        agent.save(save_path)

    # 训练结束，保存模型
    save_path = './dqn_model.ckpt'
    agent.save(save_path)

if __name__ == '__main__':
    main()
