import matplotlib.pyplot as plt

if __name__ == '__main__':
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
    best_score = 0.32
    target = [best_score] * len(index)
    init_score = [reward[0]] * len(index)

    plt.plot(index, reward)
    plt.plot(index, target)
    plt.plot(index, init_score)
    plt.axvline(x=1000, color='red', linestyle='--')
    plt.axvline(x=2000, color='red', linestyle='--')
    # 设置刻度
    plt.xticks(list(range(0, int(index[-1]), int(int(index[-1])/20))))
    plt.yticks([i * 0.2 for i in range(0, int(best_score // 0.2) + 2, 1)])
    plt.title('OLUCB')
    plt.show()

