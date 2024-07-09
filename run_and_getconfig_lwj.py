# coding: utf-8
# Author: crb
# Date: 2021/7/18 23:34
import datetime
import sys
import numpy as np
import time
from scipy import stats
import os
import subprocess
from scipy.stats.mstats import gmean
import statistics
from collections import defaultdict
import random
import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# Info:

LC_APP_NAMES     = [
                'masstree',
                'xapian'  ,
                'img-dnn' ,
                'sphinx'  ,
                'moses'   ,
                'specjbb'
                ]

# QoS requirements of LC apps (time in ms)
LC_APP_QOSES     = {
                'masstree' : 100    ,#250.0 ,
                'xapian'   : 20    ,#100.0 ,
                'img-dnn'  : 50    ,#100.0 ,
                'sphinx'   : 1500  ,#2000.0,
                'moses'    : 500    ,#1000.0,
                'specjbb'  : 10      #10
    ,#10.0
                }
# QPS levels
LC_APP_QPSES     = {
                'masstree' : list(range(200, 22000, 2000))               ,
                'xapian'   : list(range(150, 1650, 150))               ,
                'img-dnn'  : list(range(200, 4400, 400))            ,
                'sphinx'   : list(range(1, 11, 1)),
                'moses'    : list(range(20, 660, 60))             ,
                'specjbb'  : list(range(2900, 13200, 2900))
                }
# LC_APP_QPSES = {
#     'masstree': list(range(30, 330, 30)),
#     'xapian': list(range(600, 6000, 540)),
#     'img-dnn': list(range(500, 4500, 400)),
#     'sphinx': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     'moses': list(range(50, 300, 25)),
#     'specjbb': list(range(800, 8000, 720))
# }
# LC_APP_QPSES     = {
#                 'masstree' : list(range(50, 550, 50))               ,
#                 'xapian'   : list(range(80, 880, 80))               ,
#                 'img-dnn'  : list(range(140, 1540, 140))            ,
#                 'sphinx'   : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
#                 'moses'    : list(range(10, 110, 10))             ,
#                 'specjbb'  : list(range(300, 3300, 300))
#                 }
LC_APP_MAXREQS  = {
                'masstree' : list(range(200, 22000, 2000))               ,
                'xapian'   : list(range(1500, 16500, 1500))               ,
                'img-dnn'  : list(range(500, 11000, 1000))            ,
                'sphinx'   : list(range(6, 42, 6)),
                'moses'    : list(range(150, 5500, 500))             ,
                'specjbb'  : list(range(18000, 298000, 18000))
                }

# BG apps
BCKGRND_APPS  = ['blackscholes',
                'canneal',
                'fluidanimate',
                'freqmine',
                'streamcluster',
                'swaptions',
                'vips',
                'amg',
                'xsbench',
                'sw4lite'
                ]


app_docker_dict = {
        'blackscholes' : "blackscholes-lwj",
        'canneal'      : "canneal-lwj",
        'fluidanimate' : "fluidanimate-lwj",
        'freqmine'     : "freqmine-lwj",
        'streamcluster': "streamcluster-lwj",
        'swaptions'    : "swaptions-lwj"
        # 'vips'         : "vips0",
        # 'amg'          : "amg0",
        # 'xsbench'      : "xsbench0",
        # 'sw4lite'      : "sw4lite0"
      }

APP_PID = {}.fromkeys(BCKGRND_APPS, 0)
APP_PID_READERR = {}.fromkeys(BCKGRND_APPS, 0)

APP_docker_ppid = {
        'blackscholes' : "311842",
        'canneal'      : "311690",
        'fluidanimate' : "311977",
        'freqmine'     : "312133",
        'streamcluster': "312268",
        'swaptions'    : "312409",
        'vips'         : "312549",
        'amg'          : "319955",
        'xsbench'      : "321221",
        'sw4lite'      : "320905"
    }

WR_MSR_COMM       = "wrmsr -a "
RD_MSR_COMM       = "rdmsr -a -u "

# MSR register requirements
IA32_PERF_GBL_CTR = "0x38F"  # Need bits 34-32 to be 1
IA32_PERF_FX_CTRL = "0x38D"  # Need bits to be 0xFFF
MSR_PERF_FIX_CTR0 = "0x309"

# test success
def get_app_pid(app):
    # /home/pkgs/kernels/canneal/inst
    if app == 'canneal' or app == 'streamcluster':
        r = subprocess.run(f'ps aux | grep /home/pkgs/kernels/{app}/inst | grep -v grep', shell=True, check=True,
                           capture_output=True)
    else:
        r = subprocess.run(f'ps aux | grep /home/pkgs/apps/{app}/inst | grep -v grep', shell=True, check=True,
                           capture_output=True)
    r_ = str(r.stdout.decode())
    r_ = "".join(r_)
    rs = r_.split(' ')
    cnt = 0
    for m in range(len(rs)):
        if rs[m] == '':
            cnt += 1
    for x in range(cnt):
        rs.remove('')
    print(f'{app}\'s PID: {rs[1]}')
    time.sleep(0.2)
    return rs[1]

# 没用
def get_LC_app_latency_and_judge(lc_app_name):
    """
    simple version: if one app met qos,reward = -1
    complex version:
    :param lc_app_name:
    :return:
    """

    def get_lat(dir):
        with open(dir, "r") as f:
            ff = f.readlines()
            assert "latency" in ff[0], "Lat file read failed!"
            a = ff[0].split("|")[0]
            lat = a[24:-3]
            return float(lat)
    tmp =[]

    flag = 0
    for i in lc_app_name:
        dir = f'/home/pwq/atc/tailbench-v0.9/share/{i}.txt'
        while True:
            if os.path.exists(dir):
                if os.path.getsize(dir) != 0:
                    p95 = get_lat(dir)
                    tmp.append(p95)
                    break

        print('get lc ', i)
        if p95 > LC_APP_QOSES[i]:
            # qos not guaranteed
            flag = 1
    subprocess.call("sudo rm /home/pwq/atc/tailbench-v0.9/share/*.txt", shell=True)

    if flag == 1:
        return -1,tmp
    else:
        return 1,tmp


# test success
def run_bg_benchmark(bg_list,core_list):
    # 这个持久化运行
    total_command = []
    for i in range(len(bg_list)):
        print(bg_list[i])
        # 运行示例：# sudo docker exec freqmine2 taskset -c 0,1,2 python /home/run_bg.py freqmine 8
        command = f"sudo docker exec {app_docker_dict[bg_list[i]]} taskset -c {core_list[i]} python /home/run_bg.py {bg_list[i]} 8 &"
        total_command.append(command)

    subprocess.call(" ".join(total_command), shell=True, stdout= open(os.devnull, 'w'))
    print(f'start run bg app')
    time.sleep(2)
    for i in range(len(bg_list)):
        time.sleep(5)
        APP_PID[bg_list[i]] = get_app_pid(bg_list[i])
    # print(APP_PID)
        # try:
        #     try:
        #         try:
        #             try:
        #                 APP_PID[bg_list[i]] = get_app_pid(bg_list[i])
        #                 APP_PID_READERR[bg_list[i]] = 0
        #             except:
        #                 time.sleep(3)
        #                 APP_PID[bg_list[i]] = get_app_pid(bg_list[i])
        #             APP_PID_READERR[bg_list[i]] = 0
        #         except:
        #             time.sleep(3)
        #             APP_PID[bg_list[i]] = get_app_pid(bg_list[i])
        #         APP_PID_READERR[bg_list[i]] = 0
        #     except:
        #         time.sleep(3)
        #         APP_PID[bg_list[i]] = get_app_pid(bg_list[i])
        #     APP_PID_READERR[bg_list[i]] = 0
        # except:
        #     print(f'{bg_list[i]}\'s pid read error!')
        #     APP_PID_READERR[bg_list[i]] = 1

# 没用
def run_lc_benchmark(lc_list, load_list, core_list):
    # inputs:['masstree','moses','img-dnn'],[1,2,5],["0-3","4-6","7-8"]
    # notes that: the values in load_list should begin from 0
    # 这个跑一次给一次arm
    if len(lc_list) == 0:
        print('no lc app')
        return
    total_command = []
    for i in range(len(lc_list)):
        print(i,lc_list[i],load_list[i])
        qps = LC_APP_QPSES[lc_list[i]][load_list[i]]
        maxReqs = LC_APP_MAXREQS[lc_list[i]][load_list[i]]
        cores = int(core_list[i][-1]) - int(core_list[i][0]) +1
        # 运行示例：sudo docker exec img-dnn taskset -c 0,1 python /home/tailbench-v0.9/img-dnn/run_lc.py 300
        # command = f"docker exec {app_docker_dict[lc_list[i]]} taskset -c {core_list[i]} python /home/tailbench-v0.9/{lc_list[i]}/run_tail_testtt.py {qps} {cores} &"
        command = f"sudo docker exec {app_docker_dict[lc_list[i]]} taskset -c {core_list[i]} python /home/tailbench-v0.9/{lc_list[i]}/run_lc.py {qps} {maxReqs} {lc_list[i]} & "
        total_command.append(command)
    subprocess.call(" ".join(total_command), shell=True, stdout= open(os.devnull, 'w'))
    print(f'start run lc app')

def deal_features(features, num_app):
    deal_context = pd.read_csv("/home/pwq/atc/th_fair_bandit_pareto/deal_context.csv")
    deal_context = np.array(np.array(deal_context).tolist() * num_app)
    print(f'len(features): {len(features)}')
    out_dealed_features = [(features[i] - deal_context[i, 1]) / deal_context[i, 2] for i in range(len(features))]
    return out_dealed_features 

# 没用
def read_IPS_directlty(core_config):

    ipsP = os.popen("sudo " + RD_MSR_COMM + MSR_PERF_FIX_CTR0)

    # Calculate the IPS
    IPS = 0.0
    cor = [int(c) for c in core_config.split(',')]
    print("cor", cor)
    ind = 0
    for line in ipsP.readlines():
        if ind in cor:
            IPS += float(line)
        ind += 1

    return IPS

# test success
# 后台应用程序列表、核心列表、事件列表、应用程序索引、隔离 IPC 值和核心配置
# evne：性能指标列表
# app_index：一丢丢冗余
# isolated_ipc：best_ipc 测出来已知
# core_config：核分配个数
def get_now_ipc(bg_app, core_list, evne, isolated_ipc, core_config):
    now = datetime.datetime.now()
    benchmark_list = bg_app
    total_command = []
    event = ",".join(evne)
    # run_lc_benchmark(lc_app,load_list,core_list[:len(lc_app)])

    evne_tmp = []
    insn_tmp = []

    for i in range(len(bg_app)):
        if bg_app[i] == 'canneal' or bg_app[i] == 'streamcluster':
            # print('is canneal or streamcluster')
            target = f"/home/pkgs/kernels/{bg_app[i]}/"
        elif bg_app[i] == 'amg' or bg_app[i] == 'sw4lite':
            target = f"/bin/{bg_app[i]}"
        elif bg_app[i] == 'xsbench':
            target = "/bin/XSBench"
        else:
            target = f"/home/pkgs/apps/{bg_app[i]}/"
            #sudo ps aux | grep /home/pkgs/apps/fluidanimate
        cmd_run = "sudo ps aux | grep {} | grep -v grep".format(target)  # 查看当前target的进程情况
        out = os.popen(cmd_run).read()
        # print(f'out:{out}')
        if len(out.splitlines()) < 1:
            print("==============================bg app rerun")
            
            run_bg_benchmark([bg_app[i]], [core_list[i]])

        # evne_tmp记录每个app的每个指标
        # insn_tmp记录每个app每一轮的指令数
        for name in evne:
            evne_tmp.append(name + "_" + str(benchmark_list[i]))
        insn_tmp.append(f"insn per cycle_{str(benchmark_list[i])}")
        # 这里再次运行docker线程什么意思，/dev/null是linux的一个文件黑洞，任何写入的字符都将丢失
        # 似乎没什么用，绑定一次就够了
        # if APP_PID_READERR[bg_app[i]] == 0:
        #     pid = APP_PID[bg_app[i]]
        # else:
        #     pid = APP_docker_ppid[bg_app[i]]
        # subprocess.call(f'sudo taskset -apc {core_list[i]} {pid} > /dev/null',
        #                 shell=True)

        # perf测量每个正在运行的app的22个性能数据
        # sudo docker exec fluidanimate taskset -c 0,1,2 python /home/run_bg.py fluidanimate 8
        # sudo perf stat -e cycles,instructions,inst_retired.prec_dist,uops_retired.retire_slots,L1-dcache-loads,dTLB-loads,l2_rqsts.all_demand_miss,offcore_requests.demand_data_rd,dTLB-load-misses,LLC-loads,branch-misses,L1-dcache-stores,branch-loads,branch-load-misses,LLC-store-misses,cpu_clk_unhalted.thread_p,L1-dcache-load-misses,ld_blocks.store_forward,ld_blocks.no_sr -C 0,1,2 sleep 0.5
        perf_command = f"sudo perf stat -e {event} -C {core_list[i]} sleep 0.1"  # perf工具获取该app运行的22个指标
        total_command.append(perf_command)
        # print(perf_command)

    while True:
        r_ = []
        for i in range(len(benchmark_list)):
            r = subprocess.run(total_command[i], shell=True, check=True, capture_output=True)
            r_.append(str(r.stderr.decode()))
            time.sleep(0.5)
        r_ = "".join(r_)
        rs = r_.split('\n')
        label = dict.fromkeys(insn_tmp, 0)

        d = dict.fromkeys(evne_tmp, 0)

        # 用于储存究竟属于哪个app
        flag = -1
        for index, line in enumerate(rs):
            rr = line.split(' ')
            rr = [i for i in rr if i != ""]
            # print(rr)
            if len(rr) < 2 or "elapsed" in rr:
                continue
            if "Performance" in line:
                cpu = line[39:-2]
                for i in range(len(benchmark_list)):
                    if core_list[i] == cpu:
                        # 字典label记录每个app运行时核内产生的指令数
                        label[f"insn per cycle_{benchmark_list[i]}"] = float(rs[index + 3][55:59])
                        flag = benchmark_list[i]
                        break
                continue
            # 字典d记录每个app的29个性能指标
            key_name = rr[1] + "_" + str(flag)
            try:
                d[key_name] = float(rr[0].replace(",", ""))
            except:
                d[key_name] = 0.0

        # print(f'features： {list(d.values())}')
        tmp = list(d.values())
        tmp_ = deal_features(tmp, len(benchmark_list))
        tmp__ = np.array(tmp_).reshape((len(benchmark_list), 29))

        context = []
        context.append(-(tmp__.mean(axis=0)))
        context.append(tmp__.std(axis=0))
        context = np.array(context)
        context_core = context[:, 2:11].reshape((1, 18))
        context_llc = context[:, 11:20].reshape((1, 18))
        context_mb = context[:, 20:29].reshape((1, 18))
        out_context = []
        out_context.append(context_core)
        out_context.append(context_llc)
        out_context.append(context_mb)
        print('get now ipc')
        now_ipc = list(label.values())
        # 等待lc app运行完后，获取它的延时情况
        # Qos_cal,p95_list = get_LC_app_latency_and_judge(lc_app)
        print('get latency')

        for i in range(len(bg_app)):
            core = core_config[i]
            now_ipc[i] *= core

        print(isolated_ipc)
        print(now_ipc)
        speedup_list = [now_ipc[k] / isolated_ipc[k] for k in range(len(isolated_ipc))]
        fair_reward = (1 / (1 + (statistics.stdev(speedup_list) / statistics.mean(speedup_list)) ** 2))
        th_reward = gmean(speedup_list)

        print("fair_reward#################################", fair_reward)
        print("th_reward#################################", th_reward)


        if len(tmp) != 29 * len(benchmark_list) or len(now_ipc) != len(benchmark_list):
            print("ei")
            continue
        else:
            break
    times = datetime.datetime.now() -now
    print("==================================getnowipc",times)
    # print(out_context)
    print(fair_reward)
    print(th_reward)
    return out_context, fair_reward, th_reward

# 没用
def l_r_convert_config(left, right):
    if type(left) != int:
        try:
            left = int(left)
        except:
            left = int(float(left.strip('\'& \"')))
    if type(right) != int:
        try:
            right = int(right)
        except:
            right = int(float(right.strip('\'& \"')))

    bin_string = []
    for m in range(1, 29):
        if left <= m <= right:
            bin_string.append(1)
        else:
            bin_string.append(0)
    sum1 = bin_string[0] * 2 + bin_string[1]
    sum2 = bin_string[2] * 8 + bin_string[3] * 4 + bin_string[4] * 2 + bin_string[5]
    sum3 = bin_string[6] * 8 + bin_string[7] * 4 + bin_string[8] * 2 + bin_string[9]
    ans = "0x" + str(sum1) + str(hex(sum2)[-1]) + str(hex(sum3)[-1])
    return ans


# test sucess
def refer_core(core_config):
    # [2,4,3] => ["0,1","2,3,4,5","6,7,8"]
    app_cores = [""] * len(core_config)
    endpoint_left = 0
    for i in range(len(core_config)):
        endpoint_right = endpoint_left + core_config[i] - 1
        app_cores[i] = ",".join([str(c) for c in list(range(endpoint_left, endpoint_right+1))])
        endpoint_left = endpoint_right + 1
    #  将8核与9核改为10核与11核
    for i in range(len(app_cores)):
        if '8' in app_cores[i]:
            app_cores[i] = app_cores[i].replace('8', '10')
        if '9' in app_cores[i]:
            app_cores[i] = app_cores[i].replace('9', '11')
    print(app_cores)
    return app_cores

def refer_llc(llc_config):
    nof_llc = np.array(llc_config).sum()
    i = nof_llc - 1
    llc_list = []
    for j in range(len(llc_config)):
        ini_list = [0 for k in range(nof_llc)]
        count = llc_config[j]
        while count > 0:
            ini_list[i] = 1
            i -= 1
            count -= 1
        llc_list.append(hex(int(''.join([str(item) for item in ini_list]), 2)))
    return llc_list

# test success
def gen_init_config(app_id, core_arm_orders, llc_arm_orders, mb_arm_orders, NUM_UNITS,f_d_c):
    app_num = len(app_id)
    nof_core = NUM_UNITS[0]
    nof_llc = NUM_UNITS[1]
    nof_mb = NUM_UNITS[2]
    core_arms = 0
    llc_arms = 0 
    mb_arms = 0
    # 初始化core决策
    each_core_config = nof_core // app_num
    res_core_config = nof_core % app_num
    core_config = [each_core_config] * (app_num-1)
    if res_core_config >= each_core_config:
        for i in range(res_core_config):
            core_config[i]+=1
        core_config.append(each_core_config)
    else:
        core_config.append(each_core_config+res_core_config)
    # core_config = [1, 1, 1, 1, 3]
    for i in range(len(core_arm_orders)):
        if core_arm_orders[i] == core_config:
            core_arms = i
            break

    # 初始化llc决策
    each_llc_config = nof_llc // app_num
    res_llc_config = nof_llc % app_num
    llc_config = [each_llc_config] * (app_num - 1)
    if res_llc_config >= each_llc_config:
        for i in range(res_llc_config):
            llc_config[i] += 1
        llc_config.append(each_llc_config)
    else:
        llc_config.append(each_llc_config + res_llc_config)
    # llc_config = [1, 1, 1, 1, 6]
    for i in range(len(llc_arm_orders)):
        if llc_arm_orders[i] == llc_config:
            llc_arms = i
            break

    # 初始化memcached band决策
    each_mb_config = nof_mb // app_num
    res_mb_config = nof_mb % app_num
    mb_config = [each_mb_config] * (app_num - 1)
    if res_mb_config >= each_mb_config:
        for i in range(res_mb_config):
            mb_config[i] += 1
        mb_config.append(each_mb_config)
    else:
        mb_config.append(each_mb_config + res_mb_config)
    # mb_config = [1, 1, 1, 1, 6]
    for i in range(len(mb_arm_orders)):
        if mb_arm_orders[i] == mb_config:
            mb_arms = i
            break

    chosen_arms = [core_arms, llc_arms, mb_arms]
    f_d_c.writerow(core_config)
    f_d_c.writerow(llc_config)
    f_d_c.writerow(mb_config)
    f_d_c.writerow(['*', '*', '*', '*', '*', '*', '*'])
    # 将config改成具体指令
    core_list = refer_core(core_config)
    llc_list = refer_llc(llc_config)

    # 用pqos进行硬件资源隔离分区，使用core_list, llc_config, mb_config三个列表

    for i in range(len(core_config)):
        # if APP_PID_READERR[app_id[i]] == 0:
        #     pid = APP_PID[app_id[i]]
        # else:
        #     pid = APP_docker_ppid[app_id[i]]
        # subprocess.call(f'sudo taskset -apc {core_list[i]} {pid} > /dev/null', shell=True)
        subprocess.run('sudo pqos -a "llc:{}={}"'.format(i+1, core_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "llc:{}={}"'.format(i+1, llc_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -a "core:{}={}"'.format(i+1, core_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "mba:{}={}"'.format(i+1, int(float(mb_config[i])) * 10), shell=True,
                       capture_output=True)
        print(f'core{i}: {core_list[i]}')
        print(f'llc{i}: {llc_list[i]}')
        print(f'mb{i}: {int(float(mb_config[i])) * 10}')

    return core_list,llc_config,mb_config,chosen_arms, core_config


# test success
def gen_config(app_id, chosen_arms, core_arm_orders, llc_arm_orders, mb_arm_orders, f_d_c):

    core_arm, llc_arm, mb_arm = chosen_arms[0], chosen_arms[1], chosen_arms[2]
    core_config, llc_config, mb_config = [], [], []
    # core的臂选择给出一样结果，如果没有就选择第一个应用给的

    core_config = core_arm_orders[core_arm]
    llc_config = llc_arm_orders[llc_arm]
    mb_config = mb_arm_orders[mb_arm]

    core_list = refer_core(core_config)
    print(f'core_list: {core_list}')
    llc_list = refer_llc(llc_config)
    f_d_c.writerow(core_config)
    f_d_c.writerow(llc_config)
    f_d_c.writerow(mb_config)
    f_d_c.writerow(['*', '*', '*', '*', '*', '*', '*'])

# sudo taskset -apc 0 2356224 > /dev/null
    for i in range(len(core_config)):
        # sudo taskset -apc 5,6 2848052 > /dev/null
        # if APP_PID_READERR[app_id[i]] == 0:
        #     pid = APP_PID[app_id[i]]
        # else:
        #     pid = APP_docker_ppid[app_id[i]]
        # subprocess.call(f'sudo taskset -apc {core_list[i]} {pid} > /dev/null', shell=True)
        subprocess.run('sudo pqos -a "llc:{}={}"'.format(i+1, core_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "llc:{}={}"'.format(i+1, llc_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -a "core:{}={}"'.format(i+1, core_list[i]), shell=True, capture_output=True)
        subprocess.run('sudo pqos -e "mba:{}={}"'.format(i+1, int(float(mb_config[i])) * 10), shell=True, capture_output=True)
    # time.sleep(3)
    print("success!")
    return core_list, llc_config, mb_config, core_config

# 没用
def list_duplicates(choose_arm_dict, app_id):
    """
    :param seq:
    :return: multi bandit choose
    """
    core, llc, mb = [], [], []
    for i in range(len(choose_arm_dict)):
        core.append(choose_arm_dict[i][0])
        llc.append(choose_arm_dict[i][1])
        mb.append(choose_arm_dict[i][2])

    satistic_core = defaultdict(int)
    for arm_index in core:
        satistic_core[arm_index] += 1
    core_seq = sorted((satistic_core.items()))
    key, value = [], []
    for k, v in core_seq:
        key.append(k)
        value.append(v)
    core_arm = key[np.random.choice(np.where(value == np.max(value))[0])]

    satistic_llc = defaultdict(int)
    for arm_index in llc:
        satistic_llc[arm_index] += 1
    llc_seq = sorted((satistic_llc.items()))
    key, value = [], []
    for k, v in llc_seq:
        key.append(k)
        value.append(v)
    llc_arm = key[np.random.choice(np.where(value == np.max(value))[0])]

    satistic_mb = defaultdict(int)
    for arm_index in mb:
        satistic_mb[arm_index] += 1
    mb_seq = sorted((satistic_mb.items()))
    key, value = [], []
    for k, v in mb_seq:
        key.append(k)
        value.append(v)
    mb_arm = key[np.random.choice(np.where(value == np.max(value))[0])]

    choose_arm = [core_arm, llc_arm, mb_arm]
    return choose_arm

# 资源完全隔离
# u:当前递归层剩余资源数量
# r:资源种类的index
# a:当前递归层已经完成分配的app数量
def gen_configs_recursively(u, r, a, NUM_APPS, NUM_UNITS):
    if (a == NUM_APPS-1):
        return None
    else:
        ret = []
        for i in range(1, NUM_UNITS[r]-u+1-NUM_APPS+a+1):
            confs = gen_configs_recursively(u+i, r, a+1, NUM_APPS, NUM_UNITS)
            if not confs:
                ret.append([i])
            else:
                for c in confs:
                    ret.append([i])
                    for j in c:
                        ret[-1].append(j)
        return ret

# 生成所有可能的配置空间
def gen_nonOverlap_config(NUM_APPS, NUM_UNITS):
    core_config = gen_configs_recursively(0, 0, 0, NUM_APPS, NUM_UNITS)
    for i in range(len(core_config)):
        other_source = np.array(core_config[i]).sum()
        core_config[i].append(NUM_UNITS[0] - other_source)

    llc_config = gen_configs_recursively(0, 1, 0, NUM_APPS, NUM_UNITS)
    for i in range(len(llc_config)):
        other_source = np.array(llc_config[i]).sum()
        llc_config[i].append(NUM_UNITS[1] - other_source)

    mb_config = gen_configs_recursively(0, 2, 0, NUM_APPS, NUM_UNITS)
    for i in range(len(mb_config)):
        other_source = np.array(mb_config[i]).sum()
        mb_config[i].append(NUM_UNITS[2] - other_source)
    print(core_config)
    print(llc_config)
    print(mb_config)
    return core_config, llc_config, mb_config