import os
import sys
import csv
import datetime
import random
import math
import numpy as np
from run_and_getconfig_lwj import get_app_pid, run_bg_benchmark, gen_init_config, get_now_ipc, refer_core, gen_config, gen_nonOverlap_config, gen_configs_recursively

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


if __name__ == "__main__":
    NUM_APPS = 5
    NUM_UNITS = [10, 10, 10]
    core_config_list, llc_config_list, mb_config_list = gen_nonOverlap_config(NUM_APPS, NUM_UNITS)
    
    core_config_num = len(core_config_list)
    llc_config_num = len(llc_config_list)
    mb_config_num = len(mb_config_list)

    # 初始化统计数据
    # total_selections = 0
    # arm_num = core_config_num * llc_config_num * mb_config_num
    # print("arm_num:")
    # print(arm_num)
    # arm_counts = [0] * (arm_num+1)
    # arm_rewards = [0.0] * (arm_num+1)
    core_rewards = [0.0] * (core_config_num+1)
    llc_rewards = [0.0] * (llc_config_num+1)
    mb_rewards = [0.0] * (mb_config_num+1)

    round_num = 100
    app_id = ['canneal', 'fluidanimate', 'freqmine', 'streamcluster', 'blackscholes']
    f_c = open("/home/lwj/OpenOLPart/main_code/ttt.csv", "w", newline="")
    f_d_c = csv.writer(f_c)
    
    # 其他初始化操作
    isolated_ipc = [15.8, 5.6, 19.0, 12.2, 6.8]
    evne = []
    with open("/home/lwj/OpenOLPart/main_code/micro_event_we_choose_3resource.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            evne.append(line)
    init_core_list = ['0,1', '2,3', '4,5', '6,7', '8,9']

    run_bg_benchmark(app_id, init_core_list)
    core_list, llc_config, mb_config, chosen_arms, core_config = gen_init_config(app_id, core_config_list, llc_config_list, mb_config_list, NUM_UNITS,f_d_c)
    context, fair_reward, th_reward = get_now_ipc(app_id, core_list, evne, isolated_ipc, core_config)
    # print("context:")
    # print(context)

    context_core = context[0]; context_core = np.squeeze(context_core, axis=0)
    context_llc = context[1]; context_llc = np.squeeze(context_llc, axis=0)
    context_mb = context[2]; context_mb = np.squeeze(context_mb, axis=0)

    init_core_index = chosen_arms[0]+1
    init_llc_index = chosen_arms[1]+1
    init_mb_index = chosen_arms[2]+1

    core_rewards[init_core_index] += th_reward
    llc_rewards[init_llc_index] += th_reward
    mb_rewards[init_mb_index] += th_reward
    # arm_index = init_core_index * init_llc_index * init_mb_index
    # arm_rewards[arm_index] += th_reward
    # fair_reward = float(fair_reward)
    # th_reward = float(th_reward)
    f_d_c.writerow([fair_reward])
    f_d_c.writerow([th_reward])
    f_d_c.writerow(['*', '*', '*', '*', '*', '*', '*'])

    # 初始化参数
    alpha = 0.008  # LinUCB算法的alpha参数
    factor_alpha = 0.99 # alpha衰减系数

    # 获取上下文特征的维度
    n_features_core = 18
    n_features_llc = 18
    n_features_mb = 18

    # 分别初始化三个LinUCB多臂赌博机
    arm_num_core = core_config_num
    arm_num_llc = llc_config_num
    arm_num_mb = mb_config_num

    lin_ucb_core = LinUCB(alpha, arm_num_core, n_features_core)
    lin_ucb_llc = LinUCB(alpha, arm_num_llc, n_features_llc)
    lin_ucb_mb = LinUCB(alpha, arm_num_mb, n_features_mb)

    lin_ucb_core.update(init_core_index, th_reward, context_core)
    lin_ucb_llc.update(init_llc_index, th_reward, context_llc)
    lin_ucb_mb.update(init_mb_index, th_reward, context_mb)

    # 循环执行多臂赌博机算法
    for k in range(round_num):
        # 获取上下文特征
        context, fair_reward, th_reward = get_now_ipc(app_id, core_list, evne, isolated_ipc, core_config)
        # context_llc, fair_reward, th_reward = get_now_ipc(app_id, core_list, evne, isolated_ipc, llc_config)
        # context_mb, fair_reward, th_reward = get_now_ipc(app_id, core_list, evne, isolated_ipc, mb_config)

        context_core = context[0]; context_core = np.squeeze(context_core, axis=0)
        context_llc = context[1]; context_llc = np.squeeze(context_llc, axis=0)
        context_mb = context[2]; context_mb = np.squeeze(context_mb, axis=0)

        # print("########################")
        # print(context_core)
        # print(context_llc)
        # print(context_mb)
        # print("########################")

        # 使用LinUCB选择动作
        chosen_arm_core = lin_ucb_core.select_arm(context_core, factor_alpha)
        chosen_arm_llc = lin_ucb_llc.select_arm(context_llc, factor_alpha)
        chosen_arm_mb = lin_ucb_mb.select_arm(context_mb, factor_alpha)

        print("*********")
        print(chosen_arm_core)
        print(chosen_arm_llc)
        print(chosen_arm_mb)
        print("*********")

        # 解析chosen_arm到core_index, llc_index, mb_index
        core_index = chosen_arm_core
        llc_index = chosen_arm_llc
        mb_index = chosen_arm_mb

        # 执行选择的配置
        core_list, _, _, core_config = gen_config(app_id, [core_index, llc_index, mb_index], core_config_list, llc_config_list, mb_config_list, f_d_c)
        
        # 获取奖励和更新上下文
        context, fair_reward, th_reward = get_now_ipc(app_id, core_list, evne, isolated_ipc, core_config)
        # context_llc, fair_reward, th_reward = get_now_ipc(app_id, core_list, evne, isolated_ipc, core_config)
        # context_mb, fair_reward, th_reward = get_now_ipc(app_id, core_list, evne, isolated_ipc, core_config)

        context_core = context[0]; context_core = np.squeeze(context_core, axis=0)
        context_llc = context[1]; context_llc = np.squeeze(context_llc, axis=0)
        context_mb = context[2]; context_mb = np.squeeze(context_mb, axis=0)

        # 更新LinUCB模型
        lin_ucb_core.update(chosen_arm_core, th_reward, context_core)
        lin_ucb_llc.update(chosen_arm_llc, th_reward, context_llc)
        lin_ucb_mb.update(chosen_arm_mb, th_reward, context_mb)

        # 记录结果
        f_d_c.writerow([fair_reward, th_reward, core_index, llc_index, mb_index])

    f_c.close()