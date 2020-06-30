import gym
import numpy as np
import parl
from parl.utils import logger

from paddle_DDPGmodel import Model
from paddle_DDPGagent import Agent
from replay_memory import ReplayMemory

from paddle_env import Paddle
import os
from parl.algorithms import DDPG


ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate

GAMMA = 0.99      # reward 的衰减因子
TAU = 0.001       # 软更新的系数
MEMORY_SIZE = int(1e4)                  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20  # 预存一部分经验之后再开始训练
BATCH_SIZE = 1024
REWARD_SCALE = 0.1   # reward 缩放系数
NOISE = 0.05         # 动作噪声方差

TRAIN_EPISODE = 1000 # 训练的总episode数




def run_episode(agent, env, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))

        # 增加探索扰动, 输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, NOISE), 0, 2.0)
        #obs = np.squeeze(obs)
        reward, next_obs,  done= env.step(action)

        #next_obs = np.squeeze(next_obs)

        action = [action]  # 方便存入replaymemory
        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))

        if len(rpm) > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)

        obs = next_obs
        total_reward += reward

        if done or steps >= 200:
            break
    return total_reward


def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.clip(action, -1.0, 1.0)

            steps += 1
            reward, next_obs,  done = env.step(action)

            obs = next_obs
            total_reward += reward

            if render:
                env.render()
            if done or steps >= 200:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)




def main():
    # 创建环境
    env = Paddle()
    np.random.seed(0)


    act_dim = 3
    obs_dim = 5

    # 使用PARL框架创建agent

    model = Model(act_dim)
    algorithm = DDPG(model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)




    # 加载模型
    if os.path.exists('./model.ckpt'):
        save_path = './model.ckpt'
        agent.restore(save_path)
        print("模型加载成功")

    # 创建经验池
    rpm = ReplayMemory(MEMORY_SIZE)
    # 往经验池中预存数据
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(agent, env, rpm)

    episode = 0
    while episode < TRAIN_EPISODE:
        for i in range(50):
            total_reward = run_episode(agent, env, rpm)
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
