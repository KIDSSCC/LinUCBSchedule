import csv
import os
import sys
import time

import numpy as np
from LinUCB import *
from EpsilonGreedy import *

from util import *

NUM_RESOURCES_CACHE = 60  #total units of cache size


def for_epsilon_greedy():
    cm = SimulationManagement()
    curr_config = cm.receive_config()

    all_app = curr_config[0]
    num_resources = int(np.sum(curr_config[1]))
    chosen_arm = {}
    curr_allocate = curr_config[1]
    for j in range(len(all_app)):
        chosen_arm[all_app[j]] = curr_allocate[j]

    # initialize parameters
    epsilon = 0.5
    factor_alpha = 0.98
    epochs = 3000
    ucb_cache = EpsilonGreedy(epsilon, factor_alpha, len(all_app), 113)

    file_ = open('linucb.log', 'w', newline='')
    start_time = time.time()
    for i in range(epochs):
        performance = curr_config[2]
        th_reward = ucb_cache.get_now_reward(performance)
        ucb_cache.update(th_reward, chosen_arm)
        # choose an arm
        chosen_arm = ucb_cache.select_arm()
        # prepare new config
        new_config = [curr_config[0], chosen_arm]
        cm.send_config(new_config)

        # waiting for result
        curr_config = cm.receive_config()

        # write to log
        log_info = str(i) + ' ' + str(th_reward) + '\n'
        file_.write(log_info)
        if (i + 1) % 100 == 0:
            print('epoch [{} / {}]'.format(i + 1, epochs))
    end_time = time.time()
    print('used time :{}'.format(end_time - start_time))
    file_.close()


def for_linucb():
    cm = SimulationManagement()
    curr_config = cm.receive_config()

    all_app = curr_config[0]
    num_resources = int(np.sum(curr_config[1]))

    alpha = 0.90
    factor_alpha = 0.98
    n_features = 10
    epochs = 1000
    ucb_cache = LinUCB(all_app, num_resources, alpha, factor_alpha, n_features)

    file_ = open('linucb.log', 'w', newline='')
    start_time = time.time()
    for i in range(epochs):
        curr_allocate = curr_config[1]
        performance = curr_config[2]
        context_info = curr_config[3:]
        chosen_arm = {}
        for j in range(len(all_app)):
            chosen_arm[all_app[j]] = curr_allocate[j]

        th_reward = ucb_cache.get_now_reward(performance, context_info)
        ucb_cache.update(th_reward, chosen_arm)

        new_arm = ucb_cache.select_arm()
        # prepare new config
        new_config = [curr_config[0], new_arm]
        cm.send_config(new_config)

        # waiting for result
        curr_config = cm.receive_config()
        # write to log
        log_info = str(i) + ' ' + str(th_reward) + '\n'
        file_.write(log_info)
        if (i + 1) % 100 == 0:
            print('epoch [{} / {}]'.format(i + 1, epochs))


    end_time = time.time()
    print('used time :{}'.format(end_time - start_time))
    file_.close()


def for_linucb_sample():
    cm = SimulationManagement()
    curr_config = cm.receive_config()

    all_app = curr_config[0]
    num_resources = int(np.sum(curr_config[1]))

    alpha = 0.95
    factor_alpha = 0.98
    n_features = 10
    epochs = 900
    ucb_cache = LinUCB(all_app, 113, alpha, factor_alpha, n_features, True)

    file_ = open('linucb.log', 'w', newline='')
    start_time = time.time()
    file_num = 0
    for i in range(epochs):
        new_arm = ucb_cache.select_arm()
        # prepare new config
        new_config = [curr_config[0], new_arm]
        cm.send_config(new_config)

        # waiting for result
        # time.sleep(50)
        curr_config = cm.receive_config()

        curr_allocate = curr_config[1]
        performance = curr_config[2]
        context_info = curr_config[3:]
        chosen_arm = {}
        for j in range(len(all_app)):
            chosen_arm[all_app[j]] = curr_allocate[j]

        th_reward, aver_latency = ucb_cache.get_now_reward(performance, context_info)
        ucb_cache.update(th_reward, chosen_arm)

        # write to log
        log_info = 'epoch:{}: {} {} {}\n'.format(i, str(curr_allocate), th_reward, aver_latency)
        # print(log_info)
        file_.write(log_info)
        if i == 1:
            ucb_cache.save_model()  # 0号为初始模型
        if (i + 1) % 100 == 0:
            print('epoch [{} / {}]'.format(i + 1, epochs))
            ucb_cache.save_model()
            file_num +=1
        if (i + 1) % 300 == 0:
            ucb_cache.load_model(0)
        

    end_time = time.time()
    print('used time :{}'.format(end_time - start_time))
    file_.close()


if __name__ == '__main__':
    # for_epsilon_greedy()
    for_linucb_sample()
    
