import random
import time
from multiprocessing import Process, Manager
import numpy as np
import itertools

class LinUCB:
    def __init__(self, alpha, n_arms, n_features):
        self.alpha = alpha
        self.n_arms = n_arms
        self.n_features = n_features
        self.A = np.array([np.identity(n_features) for _ in range(n_arms)])
        self.b = np.array([np.zeros(n_features) for _ in range(n_arms)])

    def select_arm(self, context, factor_alpha):
        p = np.zeros(self.n_arms)
        self.alpha *= factor_alpha
        
        # print("n_arms:")
        # print(self.n_arms)

        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = np.dot(A_inv, self.b[arm])
            p[arm] = np.dot(theta.T, context) + self.alpha * np.sqrt(np.dot(context.T, np.dot(A_inv, context)))
        return np.argmax(p)

    def update(self, chosen_arm, reward, context):
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context

def gen_all_config(num_apps, num_resources):
    '''
    generate all resource config according to the number of apps and total resources

    Args:
        num_apps (int): number of apps
        num_resources (int): total units of resources

    Returns:
        list<list>: a list containing all possible config, which is list<int>
    '''
    if num_apps == 1:
        # Only one app, it get all remaining resources
        return [[num_resources]]
    
    all_config = []
    for i in range(num_resources + 1):
        # Recursively allocate the remaining resources among the remaining app
        for sub_allocation in gen_all_config(num_apps - 1, num_resources - i):
            all_config.append([i] + sub_allocation)
    return all_config

def get_now_reward(curr_metrics):
    '''
    Get a default context and average reward

    Args:
        curr_metrics (list<float>): a feedback metric representing the current mixed deployment status for each app

    Returns:
        list<float>: context infomation, initialize as a list of 18 elements, each set to 1.0
        float: th_reward, the average reward calculated based on the current metrics
    '''
    context = [1.0 for _ in range(18)]
    th_reward = sum(float(x) for x in curr_metrics)/len(curr_metrics)
    return context, th_reward

def find_vectors(x, d_max):
    n = len(x)
    S = sum(x)
    solutions = []

    def backtrack(current_vector, index, current_distance):
        if current_distance > d_max or any(val < 0 for val in current_vector):
            return
        if index == n:
            if sum(current_vector) == S:
                solutions.append(list(current_vector))
            return
        for delta in range(-d_max, d_max + 1):
            current_vector[index] += delta
            new_distance = current_distance + abs(delta)
            backtrack(current_vector, index + 1, new_distance)
            current_vector[index] -= delta

    initial_vector = x.copy()
    backtrack(initial_vector, 0, 0)
    return solutions


class EpsilonGreedy:
    def __init__(self, epsilon, factor_alpha, num_apps, num_resources):
        self.init_factor = [epsilon, factor_alpha, 0.95, 0.98]
        self.epsilon = self.init_factor[0]
        self.factor_alpha = self.init_factor[1]
        self.all_cache_config = gen_all_config(num_apps, num_resources)
        self.n_arms = len(self.all_cache_config)

        # 探索过程随机探索与邻近探索的比例
        self.sub_epsilon = self.init_factor[2]
        self.sub_alpha = self.init_factor[3]
        # 邻近探索的曼哈顿距离
        self.dmax = 10

        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.neighbour = {}
        self.last_reward = [-1] * self.n_arms

        # 并行加速搜索临近配置的过程
        st_time = time.time()
        # for i in range(self.n_arms):
        #     self.neighbour[i] = find_vectors(self.all_cache_config[i], self.dmax)
        all_threads = []
        parts_neighbour = []
        num_tasks = self.n_arms // 10
        with Manager() as manager:
            parts_neighbour.append(manager.dict())
            for i in range(0, self.n_arms, num_tasks):
                all_threads.append(Process(target=self.find_neighbours, args=(i, min(i + num_tasks, self.n_arms), parts_neighbour[-1])))
                all_threads[-1].start()
            for i in range(len(all_threads)):
                all_threads[i].join()
            for i in range(len(parts_neighbour)):
                self.neighbour.update(parts_neighbour[i])
        en_time = time.time()
        print('construct neighbour :{}'.format(en_time - st_time))

    def find_neighbours(self, start, end, part_neighbour):
        # 并行执行
        start_time = time.time()
        for i in range(start, end):
            part_neighbour[i] = find_vectors(self.all_cache_config[i], self.dmax)
        end_time = time.time()
        # print('----- find_neighbour, from {} to {}, Used Time: {}'.format(start, end, end_time - start_time))

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            # 探索
            self.epsilon *= self.factor_alpha
            if np.random.rand() < self.sub_epsilon:
                # 随机探索
                self.sub_epsilon *=self.sub_alpha
                chosen_arm = np.random.randint(self.n_arms)
            else:
                # 最优点临近探索
                curr_max = np.argmax(self.values)
                num_neighbours = len(self.neighbour[int(curr_max)])
                chosen = random.randint(0, num_neighbours - 1)
                chosen_arm = self.all_cache_config.index(self.neighbour[int(curr_max)][chosen])
        else:
            # 利用
            chosen_arm = np.argmax(self.values)

        return chosen_arm, self.all_cache_config[chosen_arm]

    def update(self, chosen_arm_index, reward):
        last_reward = self.last_reward[chosen_arm_index]
        if last_reward > 0 and float(abs(reward - last_reward)) / last_reward > 0.05:
            # workload change
            print('-----judge workload change, {}'.format(float(abs(reward - last_reward)) / last_reward))
            self.last_reward = [-1] * self.n_arms
            self.counts = np.zeros(self.n_arms)
            self.values = np.zeros(self.n_arms)
            self.epsilon = self.init_factor[0]
            self.sub_epsilon = self.init_factor[2]

        self.counts[chosen_arm_index] += 1
        n = self.counts[chosen_arm_index]
        value = self.values[chosen_arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm_index] = new_value
        self.last_reward[chosen_arm_index] = reward


if __name__ == '__main__':
    num_tasks = 29055 // 10 if 29055 % 10 == 0 else 29055 // 10 + 1
    print(num_tasks)