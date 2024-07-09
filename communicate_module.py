import socket
import pickle
import os
import sys
import csv
import datetime
import random
import math
import numpy as np
from run_and_getconfig_cache_lwj import get_app_pid, run_bg_benchmark, gen_init_cache_config, get_now_reward, refer_llc, gen_config, gen_nonOverlap_config, gen_configs_recursively

class LinUCB:
    def __init__(self, alpha, n_arms, n_features):
        self.alpha = alpha
        self.n_arms = n_arms
        self.n_features = n_features
        # self.factor_alpha = factor_alpha
        # self.reward = reward
        self.A = np.array([np.identity(n_features) for _ in range(n_arms)])
        self.b = np.array([np.zeros(n_features) for _ in range(n_arms)])

    def select_arm(self, context, factor_alpha):
        p = np.zeros(self.n_arms)
        self.alpha *= factor_alpha
        # print("P=")
        # print(p)
        
        # print("n_arms:")
        print(self.n_arms)

        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = np.dot(A_inv, self.b[arm])
            
            # print("A:"); print(A_inv)
            # print("theta:"); print(theta)
            
            p[arm] = np.dot(theta.T, context) + self.alpha * np.sqrt(np.dot(context.T, np.dot(A_inv, context)))

            # print("p[arm]"); print(p[arm])
        return np.argmax(p)

    def update(self, chosen_arm, reward, context):
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context


'''
receive the current config from bench
'''
def receive_config():
    # create the server socket
    print("receive config")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('127.0.0.1', 1412))
    server_socket.listen(1)
    
    # listening the old_config from the bench
    client_socket, client_address = server_socket.accept()
    received_data = client_socket.recv(1024)
    config = pickle.loads(received_data)
    
    client_socket.close()
    server_socket.close()
    return config


'''
send the new config to bench
'''
def send_config(new_config):
    serialized_config = pickle.dumps(new_config)

    # connect to the bench
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 1413))
    client_socket.send(serialized_config)
    client_socket.close()
    print("send_config success!")
    return


if __name__ == '__main__':
    print("111")
    old_config = receive_config()
    print("old_config: ")
    print(old_config)
    '''
    eg:
        old_config[0]: ['uniform', 'zipfian']    # name of pool and the turn of pool size and latency
        old_config[1]: [16, 16]                  # current pool size,64MB per unit, for example, 16 means current pool size is 1G
        old_config[2]: [0.0321, 0.0234]          # latency of each workload
        old_config[3]: [0.8254, 0.7563]          # cache hit rate of each workload
    '''
    NUM_APPS = len(old_config[0])
    NUM_UNITS = [30, 10, 10]
    cache_config_list, llc_config_list, mb_config_list = gen_nonOverlap_config(NUM_APPS, NUM_UNITS)
    print("cache_config_list: ")
    print(cache_config_list)

    cache_config_num = len(cache_config_list)
    llc_config_num = len(llc_config_list)
    mb_config_num = len(mb_config_list)
    print("**********************")
    print(cache_config_list[225])
    print("**********************")

    round_num = 200
    f_c = open("/home/md/SHMCachelib/RemoteSchedule/ttt.csv", "w", newline="")
    f_d_c = csv.writer(f_c)

    # 其他初始化操作
    init_core_list = ['0,1', '2,3', '4,5', '6,7', '8,9']

    # 初始化运行程序
    # run_bg_benchmark(app_id, init_core_list) 

    # 生成初始config（公平分配）
    app_id = old_config[0]
    cache_config = old_config[1]
    cache_reward = old_config[2]
    cache_m = old_config[3]
    init_cache_index = cache_config_list.index(cache_config)

    # init_config = []
    # init_config.append(old_config[0])
    # random_cache_index = random.randint(0, cache_config_num-1)
    # init_partition = cache_config_list[random_cache_index]
    # init_config.append(init_partition)

    # send_config(init_config)
    # new_config = receive_config()
    # cache_reward = new_config[2]

    # 上下文+reward（暂时简单定义）
    context, th_reward = get_now_reward(cache_m)
    # context, th_reward = get_now_reward(old_cache_reward, new_cache_reward)
    print("context")
    print(context)
    print("th_reward")
    print(th_reward)

    # 初始化参数
    alpha = 0.4  # LinUCB算法的alpha参数
    factor_alpha = 0.994 # alpha衰减系数

    # 获取上下文特征的维度
    n_features_cache = 18
    # context_cache = context.tolist()
    context_cache = np.array(context)
    #context_cache = np.squeeze(context_cache, axis=0)

    arm_num_cache = cache_config_num
    print("arm_num_cache")
    print(arm_num_cache)
    print("====================")
    lin_ucb_cache = LinUCB(alpha, arm_num_cache, n_features_cache)
    lin_ucb_cache.update(init_cache_index, th_reward, context_cache)

    # for i in range(10):
    #     exp_config = []
    #     exp_config.append(old_config[0])
    #     exp_partition = [6,18,6]
    #     exp_config.append(exp_partition)
    #     send_config(exp_config)
    #     get_config = receive_config()
    #     print("------------------------")
    #     print(get_config[0])
    #     print(get_config[1])
    #     print(get_config[2])
    #     print(get_config[3])
    #     _, t_reward = get_now_reward(get_config[3])
    #     print(t_reward)
    #     print("-------------------------")

    # for i in range(10):
    #     exp_config = []
    #     exp_config.append(old_config[0])
    #     exp_partition = [10,10,10]
    #     exp_config.append(exp_partition)
    #     send_config(exp_config)
    #     get_config = receive_config()
    #     print("========================")
    #     print(get_config[0])
    #     print(get_config[1])
    #     print(get_config[2])
    #     print(get_config[3])
    #     _, t_reward = get_now_reward(get_config[3])
    #     print(t_reward)
    #     print("=========================")

    # 循环执行多臂赌博机算法
    for k in range(round_num):
        # 获取上下文特征
        context, th_reward = get_now_reward(cache_m)
        # context, th_reward = get_now_reward(old_cache_reward, new_cache_reward)
        # old_cache_reward = new_cache_reward
        # context_cache = context.tolist()
        context_cache = np.array(context)
        #context_cache = np.squeeze(context_cache, axis=0)

        # 使用LinUCB选择动作
        chosen_arm_cache = lin_ucb_cache.select_arm(context_cache, factor_alpha)

        print("*********")
        print(chosen_arm_cache)
        print("*********")

        # 解析chosen_arm到core_index, llc_index, mb_index
        cache_index = chosen_arm_cache
        new_config = []
        new_config.append(old_config[0])
        new_partition = cache_config_list[cache_index]
        new_config.append(new_partition)

        # new_config = cache_config_list
        # 执行选择的配置
        send_config(new_config)
        old_config = receive_config()

        app_id = old_config[0]
        cache_config = old_config[1]
        cache_reward = old_config[2]
        cache_m = old_config[3]

        # 获取奖励和更新上下文
        context, th_reward = get_now_reward(cache_m)
        # context, th_reward = get_now_reward(old_cache_reward, new_cache_reward)
        
        # context_cache = context.tolist()
        context_cache = np.array(context)
        #context_cache = np.squeeze(context_cache, axis=0)

        # 更新LinUCB模型
        lin_ucb_cache.update(chosen_arm_cache, th_reward, context_cache)
        print("*********---------")
        print("th_reward:")
        print(th_reward)
        print("*********---------")
        # 记录结果
        f_d_c.writerow([th_reward, cache_index])

    f_c.close()


    # new_config = []
    # new_config.append(old_config[0])
    # new_partition = [old_config[1][0]-2, old_config[1][1]+2]
    # new_config.append(new_partition)

    # '''
    # eg:
    #     new_config[0]: ['uniform', 'zipfian']   # be the same as old_config[0]
    #     new_config[1]: [10, 22]                 # the new cache allocation after the resource tuning
    # '''
    # send_config(new_config)
