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
from parl.algorithms import DDPG
from make_test_env import *
from operator import add


LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.005
GAMMA = 0.99  # discount factor of reward
TRAIN_EPISODE = 100  # 训练的总episode数

epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

# generate_routefile_random(episode_time, num_vehicles)
# generate_routefile(290,10)
traci.start([sumoBinary, "-c", "data/cross.sumocfg",
             "--tripinfo-output", "tripinfo.xml"])

traci.trafficlight.setPhase("0", 0)

nA = 2


def test(agent):


    traci.load(["--start", "-c", "data/cross.sumocfg",
                "--tripinfo-output", "tripinfo.xml"])
    traci.trafficlight.setPhase("0", 0)

    obs, _, _ = getState_baseline(transition_time)
    counter = 0
    stride = 0

    length_data_avg = []
    count_data = []
    delay_data_avg = []
    delay_data_min = []
    delay_data_max = []
    delay_data_time = []
    current_left_time = 0
    current_top_time = 0
    current_bottom_time = 0
    current_right_time = 0
    overall_lane_qlength = [0, 0, 0, 0]
    num_cycles = 0
    num_qlength_instances = 0
    total_t = 0
    sum_q_lens = 0
    while traci.simulation.getMinExpectedNumber() > 0:

        # print("Waiting time on lane 1i_0 = ",getWaitingTime("1i_0"))

        print("Inside episode counter", counter)

        counter += 1
        total_t += 1
        # batch_experience = experience[:batch_history]
        prev_phase = traci.trafficlight.getPhase("0")

        action = agent.predict(obs.astype('float32'))


        print("action:",action)
        # queueLength = getQueueLength()
        next_obs, qlength, avg_lane_qlength = makeMove(action, transition_time)
        new_phase = traci.trafficlight.getPhase("0")
        print("Previous phase = ", prev_phase)
        print("New phase = ", new_phase)
        vehicleList = traci.vehicle.getIDList()
        num_vehicles = len(vehicleList)
        print("Number of cycles = ", num_cycles)

        if num_vehicles:
            avg = 0
            max = 0
            mini = 100
            for id in vehicleList:
                time = traci.vehicle.getAccumulatedWaitingTime(id)
                if time > max:
                    max = time

                if time < mini:
                    mini = time

                avg += time
            avg /= num_vehicles
            delay_data_avg.append(avg)
            delay_data_max.append(max)
            delay_data_min.append(mini)
            length_data_avg.append(qlength)
            count_data.append(num_vehicles)
            delay_data_time.append(traci.simulation.getCurrentTime() / 1000)

            if traci.simulation.getCurrentTime() / 1000 < 2100:
                overall_lane_qlength = list(map(add, overall_lane_qlength, avg_lane_qlength))
                num_qlength_instances += 1
                if prev_phase == 3 and new_phase == 0:
                    num_cycles += 1
                if prev_phase == 0:
                    current_bottom_time += transition_time
                if prev_phase == 1:
                    current_right_time += transition_time
                if prev_phase == 2:
                    current_top_time += transition_time
                if prev_phase == 3:
                    current_left_time += transition_time

        obs = next_obs

    overall_lane_qlength[:] = [x / num_qlength_instances for x in overall_lane_qlength]
    current_right_time /= num_cycles
    current_top_time /= num_cycles
    current_left_time /= num_cycles
    current_bottom_time /= num_cycles
    avg_free_time = [current_left_time, current_top_time, current_right_time, current_bottom_time]

    plt.plot(delay_data_time, delay_data_avg, 'b-', label='avg')
    # plt.plot(delay_data_time, delay_data_min, 'g-', label='min')
    # plt.plot(delay_data_time, delay_data_max,'r-', label='max')
    plt.legend(loc='upper left')
    plt.ylabel('Waiting time per minute')
    plt.xlabel('Time in simulation (in s)')

    plt.figure()
    plt.plot(delay_data_time, length_data_avg, 'b-', label='avg')
    plt.legend(loc='upper left')
    plt.ylabel('Average Queue Length')
    plt.xlabel('Time in simulation (in s)')

    plt.figure()
    plt.plot(delay_data_time, count_data, 'b-', label='avg')
    plt.legend(loc='upper left')
    plt.ylabel('Average Number of Vehicles in Map')
    plt.xlabel('Time in simulation (in s)')

    plt.figure()
    label = ['Obstacle Lane', 'Top Lane w/ traffic', 'Right lane', 'Bottom lane']
    index = np.arange(len(label))
    plt.bar(index, avg_free_time, color=['red', 'green', 'blue', 'blue'])
    plt.xlabel('Lane')
    plt.ylabel('Average Green Time per Cycle (in s)')
    plt.xticks(index, label)
    axes = plt.gca()
    axes.set_ylim([0, 60])

    plt.figure()
    label = ['Obstacle Lane', 'Top Lane w/ traffic', 'Right lane', 'Bottom lane']
    index = np.arange(len(label))
    plt.bar(index, overall_lane_qlength, color=['red', 'green', 'blue', 'blue'])
    plt.xlabel('Lane')
    plt.ylabel('Average Q-length every 8 seconds')
    plt.xticks(index, label)
    axes = plt.gca()
    axes.set_ylim([0, 20])
    plt.show()

    AVG_Q_len_perepisode.append(sum_q_lens / 702)
    sum_q_lens = 0


if __name__ == '__main__':
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

    test(agent)
