import gym
import numpy as np
import parl
from parl.utils import logger

from paddle_model import Model
from paddle_agent import Agent

from replay_memory import ReplayMemory

from paddle_env import Paddle


LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 1024   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等



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
    env = Paddle()
    np.random.seed(0)

    action_dim = 3
    obs_shape = 5

    model = Model(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(
        model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_shape,
        act_dim=action_dim,
        e_greed=0.1,  # explore
        e_greed_decrement=1e-6
    )  # probability of exploring is decreasing during training

    save_path = './dqn_model.ckpt'
    agent.restore(save_path)
    print("模型加载成功")
    eval_reward = evaluate(agent, env)
    logger.info('test_reward:{}'.format(eval_reward))



if __name__ == '__main__':
    main()