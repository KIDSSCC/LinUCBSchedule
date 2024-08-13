import matplotlib.pyplot as plt
import re

def simulation():
    log_name = 'linucb.log'
    index = []
    reward = []
    count = 1
    with open(log_name, 'r') as log_file:
        for line in log_file.readlines():
            idx, rew = line.split(' ')
            index.append(idx)
            reward.append(rew)
    reward = [float(i) for i in reward]
    best_score = 0.579
    target = [best_score] * len(index)

    init_score = [reward[0]] * len(index)

    plt.plot(index, reward)
    plt.plot(index, target)
    plt.plot(index, init_score)
    plt.axvline(x=1000, color='red', linestyle='--')
    plt.axvline(x=2000, color='red', linestyle='--')
    # 设置刻度
    plt.xticks(list(range(0, int(index[-1]), int(int(index[-1]) / 20))))
    plt.yticks([i * 0.2 for i in range(0, int(best_score // 0.2) + 2, 1)])
    plt.title('OLUCB')
    plt.show()


def proto_system():
    log_name = '0813_2.log'
    index = []
    reward = []
    count = 1
    with open(log_name, 'r') as log_file:
        for line in log_file.readlines():
            line = line.rstrip('\n')
            token = re.split(r'[: ]+', line)
            index.append(token[1])
            reward.append(token[-1])
    reward = [float(i) for i in reward]
    result = reward[-10:]
    result = sum(result) / len(result)
    baseline_time = 460.76
    print('average result: {}'.format(result))
    print('lifting range: {}'.format((baseline_time - result) / baseline_time) )
    baseline = [baseline_time] * len(index)
    plt.plot(index, reward, label='OLUCB')
    plt.plot(index, baseline, label='baseline')
    # 设置刻度
    plt.xticks(list(range(0, int(index[-1]) + 1, int(int(index[-1]) / 20))))
    plt.yticks(list(range(200, int(max(reward) + 20), 20)))
    plt.title('ProtoSystem Colocation')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    proto_system()
