import gym
import numpy as np
import parl
from parl.utils import logger

from jump_PGmodel import Model
from jump_PGagent import Agent

from jump_env import JumpGame
import os
from parl.algorithms import PolicyGradient


LEARNING_RATE = 0.001

def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    #obs = np.reshape(obs, (1, 13))
    obs = np.squeeze(obs)
    while True:
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        reward, obs, done = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list

def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()

        episode_reward = 0
        while True:
            action = agent.predict(obs) # 选取最优动作
            reward, obs, isOver= env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)



def main():
    env = JumpGame()
    np.random.seed(0)


    action_dim = 2
    obs_shape = 13



    model = Model(act_dim=action_dim)
    algorithm = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(algorithm, obs_dim=obs_shape, act_dim=action_dim)




    # 加载模型
    if os.path.exists('./model.ckpt'):
        save_path = './model.ckpt'
        agent.restore(save_path)
        print("模型加载成功")

    for i in range(1000):
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 10 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = evaluate(env, agent, render=False)  # render=True 查看渲染效果，需要在本地运行，AIStudio无法显示
            logger.info('Test reward: {}'.format(total_reward))
            save_path = './model/dqn_model_{}_{}.ckpt'.format(i, total_reward)
            agent.save(save_path)

    # 保存模型到文件 ./model.ckpt
    agent.save('./model.ckpt')


if __name__ == '__main__':
    main()
