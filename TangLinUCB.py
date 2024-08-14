import random
from itertools import zip_longest
from ScheduleFrame import *
import pickle
import os

def gen_feasible_configs(num_of_cache, cache_top_k):
    # TODO：根据各个应用的top_k整理出的所有可行的结果
    num_app = len(cache_top_k)
    top_k = len(cache_top_k[0])

    def gen_side(tmp, k, n=1):
        """
        :param root: root node, first app node
        :param n: n_th in top_k
        :return:
        """
        if n == num_app:
            return [[]]
        for k in range(top_k):
            t1 = k * top_k ** (num_app - n - 1)
            t2 = top_k ** (num_app - 1) - (top_k - k - 1) * top_k ** (num_app - n - 1)
            delta2 = top_k ** (num_app - 1 - n)
            delta1 = top_k ** (num_app - n)
            for i in range(t1, t2, delta1):
                for j in range(i, i + delta2):
                    app_core = cache_top_k[n][k]
                    feasible_core = num_of_cache - sum(tmp[j])
                    if app_core > feasible_core:
                        tmp[j].append(feasible_core)
                    else:
                        tmp[j].append(app_core)

        gen_side(n=n + 1, tmp=tmp, k=k)
        return tmp

    all_feasible_config = []
    for i in range(len(cache_top_k)):
        cache_top_k.append(cache_top_k.pop(0))
        for j in range(top_k):
            tmp = [[cache_top_k[0][j]] for _ in range(top_k ** (num_app - 1))]
            res = gen_side(tmp, k=0)
            for k in range(i + 1):
                res = [[row[-1]] + row[:-1] for row in res]
            all_feasible_config.extend(res)

    # 先去重
    unique_tuples = set(tuple(x) for x in all_feasible_config)
    all_feasible_config = [list(x) for x in unique_tuples]
    # 对未利用的资源重新分配
    for config in all_feasible_config:
        assert sum(config) <= num_of_cache, 'The allocated cache exceeds the limit'
        if sum(config) < num_of_cache:
            left_cache = num_of_cache - sum(config)
            for i in range(left_cache):
                min_value = min(config)
                min_index = config.index(min_value)
                config[min_index] += 1
    return all_feasible_config


def get_top_k(arr, k, times):
    if times < 5 or random.randint(1, 10) > 8:
        arr_top_k_id = [random.randint(0, len(arr) - 1) for _ in range(k)]
    else:
        arr_top_k_id = np.argsort(arr)[-k:]
    return list(arr_top_k_id)


def beam_search(num_of_cache, all_apps, p_c_t, times, end_condition=30):
    # TODO：从各个子bandit中选出top_k的配置，并进行组合，形成全局的配置
    action = {}.fromkeys(all_apps)
    num_app = len(all_apps)
    top_k = int(10 ** (np.log10(end_condition) / num_app))

    cache_top_k = [get_top_k(p_c_t[all_apps[i]], top_k, times) for i in range(num_app)]
    feasible_configs = gen_feasible_configs(num_of_cache=num_of_cache, cache_top_k=cache_top_k)
    sum_p_list = []
    for config in feasible_configs:
        try:
            config_p = [p_c_t[all_apps[i]][config[i]] for i in range(num_app)]
            sum_p = sum(config_p)
            sum_p_list.append(sum_p)
        except IndexError as e:
            print(f"Caught an IndexError: {e}")
            print(config)
            print(cache_top_k)
    cache_act = feasible_configs[np.argmax(sum_p_list)]
    for i in range(num_app):
        action[all_apps[i]] = cache_act[i]
    return action


def latin_hypercube_sampling(n, d, m, M, ratio):
    # Initialize the samples array
    samples = np.zeros((n, d))
    for i in range(n):
        # Generate d random numbers and normalize them
        while True:
            x = np.random.uniform(m, M, d)
            if np.sum(x) >= M:
                break
        x = x / np.sum(x) * M

        # Check if all elements are in the range [m, M]
        if np.all(x >= m) and np.all(x <= M):
            samples[i] = x
        else:
            # Re-generate the sample if it doesn't satisfy the constraints
            i -= 1
    sample_config = []
    for i in range(len(samples)):
        tmp = [int(ele) for ele in samples[i]]
        curr_sum = sum(tmp)
        tmp[-1] += M - curr_sum
        sample_config.append([ele * ratio for ele in tmp])
    return sample_config

def print_model_state(model):
    print("Alpha:", model.alpha)
    print("Times:", model.times)
    # print("Sampling model:", model.sampling_model)
    # print("Sample result:", model.sample_result)
    # print("Sample config:", model.sample_config)
    print("Current best config:", model.curr_best_config)
    print("Current best reward:", model.curr_best_reward)
    # print("Duration period:", model.duration_period)
    # print("History reward:", model.history_reward)
    
    # print("A_c:", model.A_c)
    # print("b_c:", model.b_c)
    # print("p_c_t:", model.p_c_t)
    # print("Context:", model.context)
    # print("Other context:", model.other_context)

class LinUCB(ScheduleFrame):
    def __init__(self, all_apps, n_cache, alpha, factor_alpha, n_features, sample=False):
        super().__init__()
        self.all_apps = all_apps
        self.num_apps = len(all_apps)
        self.n_arms = n_cache + 1
        self.n_features = n_features

        self.init_factor = [alpha, factor_alpha]
        self.alpha = self.init_factor[0]
        self.factor_alpha = self.init_factor[0]

        self.times = 0
        self.sampling_model = sample
        self.sample_result = []
        self.sample_config = None
        self.sample_times = 10
        self.ratio = 10

        self.curr_best_config = None
        self.curr_best_reward = None
        self.duration_period = 0
        self.threshold = 150
        self.probability_threshold = 90

        self.save_model_num = 0 # 用于保存模型

        self.history_reward = []

        self.A_c = {}
        self.b_c = {}
        self.p_c_t = {}

        for app in self.all_apps:
            self.A_c[app] = np.zeros((self.n_arms, self.n_features * 2, self.n_features * 2))
            self.b_c[app] = np.zeros((self.n_arms, self.n_features * 2, 1))
            self.p_c_t[app] = np.zeros(self.n_arms)
            for arm in range(self.n_arms):
                self.A_c[app][arm] = np.eye(self.n_features * 2)

        # contexts
        self.context = {}
        self.other_context = {}
        sum = np.zeros(n_features)
        for app in self.all_apps:
            self.context[app] = [1.0 for _ in range(self.n_features)]
            sum += np.array(self.context[app])
        for app in self.all_apps:
            self.other_context[app] = list((sum - np.array(self.context[app])) / (len(self.all_apps) - 1))

        # Hypercube Sampling
        if self.sampling_model:
            self.sample_config = latin_hypercube_sampling(self.sample_times, self.num_apps, 0, (self.n_arms - 1)//self.ratio, self.ratio)
        return

    def select_arm(self):
        if self.sampling_model:
            if self.times == len(self.sample_config):
                # sample end
                self.times += 1
                self.sampling_model = False
                # print(max(self.sample_result))
                return self.sample_config[self.sample_result.index(max(self.sample_result))]
            # initial phase
            cache_allocation = self.sample_config[self.times]
            self.times += 1
            return cache_allocation

        self.times += 1
        if self.duration_period > self.threshold and random.randint(1, 100) < self.probability_threshold:
            cache_allocation = []
            for app in self.all_apps:
                cache_allocation.append(self.curr_best_config[app])
            return cache_allocation
        else:
            contexts = {}
            for app in self.all_apps:
                A = self.A_c[app]
                b = self.b_c[app]
                contexts[app] = np.hstack((self.context[app], self.other_context[app]))

                for i in range(self.n_arms):
                    theta = np.linalg.inv(A[i]).dot(b[i])
                    cntx = np.array(contexts[app])
                    self.p_c_t[app][i] = theta.T.dot(cntx) + self.alpha * np.sqrt(
                        cntx.dot(np.linalg.inv(A[i]).dot(cntx)))
            cache_action = beam_search(self.n_arms - 1, self.all_apps, self.p_c_t, self.times, end_condition=30)

            # cache_action is a dict
            cache_allocation = []
            for app in self.all_apps:
                cache_allocation.append(cache_action[app])

            self.alpha *= self.factor_alpha
            return cache_allocation

    def update(self, reward, chosen_arm):
        if self.sampling_model:
            # initial phase
            self.sample_result.append(float(reward))
        else:
            if self.curr_best_config is None:
                self.curr_best_config = chosen_arm
                self.curr_best_reward = reward
                self.duration_period = 1
                # self.history_reward = []
            else:
                if reward > self.curr_best_reward:
                    self.curr_best_reward = reward
                    self.curr_best_config = chosen_arm
                    self.duration_period = 1
                    # self.history_reward = []
                else:
                    self.duration_period += 1

            contexts = {}
            for app in self.all_apps:
                arm = chosen_arm[app]
                contexts[app] = np.hstack((self.context[app], self.other_context[app]))
                self.A_c[app][arm] += np.outer(np.array(contexts[app]), np.array(contexts[app]))
                self.b_c[app][arm] = np.add(self.b_c[app][arm].T, np.array(contexts[app]) * reward).reshape(self.n_features * 2, 1)

            if self.times > 200:
                if len(self.history_reward) < 60:
                    self.history_reward.append(round(float(reward), 3))
                else:
                    first_half = self.history_reward[:30]
                    second_half = self.history_reward[30:]
                    first_aver = sum(first_half) / len(first_half)
                    second_aver = sum(second_half) / len(second_half)
                    # print('{} --> {},  {}'.format(first_aver, second_aver, abs(second_aver - first_aver) / first_aver))
                    if abs(second_aver - first_aver) / first_aver > 0.20:
                        # workload change
                        print('----- test workload change -----')
                        self.reset()
                    else:
                        self.history_reward.pop(0)

    def get_now_reward(self, performance, context_info=None):
        # update the context
        tmp = [list(row) for row in zip_longest(*context_info, fillvalue=None)]
        sum_context = np.zeros(self.n_features)
        for i, app in enumerate(self.all_apps):
            self.context[app] = tmp[i]
            sum_context += np.array(self.context[app])
        for app in self.all_apps:
            self.other_context[app] = list((sum_context - np.array(self.context[app])) / (len(self.all_apps) - 1))

        # calculate the reward for hitrate etc bigger is greater
        th_reward = sum(float(x) for x in performance) / len(performance)
        # smaller is greater
        # aver_latency = sum(float(x) for x in performance) / len(performance)
        # th_reward = 100 / aver_latency
        aver_latency = 0
        return th_reward, aver_latency

    def reset(self):
        self.alpha = self.init_factor[0]
        self.times = 0
        self.sampling_model = True
        self.sample_config = latin_hypercube_sampling(20, self.num_apps, 0, (self.n_arms - 1)//self.ratio, self.ratio)
        self.sample_result = []

        self.curr_best_config = None
        self.curr_best_reward = None
        self.duration_period = 0

        self.history_reward = []

        for app in self.all_apps:
            self.A_c[app] = np.zeros((self.n_arms, self.n_features * 2, self.n_features * 2))
            self.b_c[app] = np.zeros((self.n_arms, self.n_features * 2, 1))
            self.p_c_t[app] = np.zeros(self.n_arms)
            for arm in range(self.n_arms):
                self.A_c[app][arm] = np.eye(self.n_features * 2)

    def save_model(self):
        file_name = 'train_dir/save_model' + str(self.save_model_num) + '.pkl'
        self.save_model_num += 1
        print('save model',file_name)
        state = {
            'all_apps' : self.all_apps,
            'num_apps' : self.num_apps,
            'n_arms' : self.n_arms,
            'n_features' : self.n_features,
            'init_factor' : self.init_factor,
            'alpha' : self.alpha ,
            'factor_alpha' : self.factor_alpha,
            'times' : self.times,
            'sampling_model' : self.sampling_model,
            'sample_result' : self.sample_result,
            'sample_config' : self.sample_config,
            'sample_times' : self.sample_times,
            'ratio' : self.ratio,
            'curr_best_config' : self.curr_best_config,
            'curr_best_reward' : self.curr_best_reward,
            'duration_period' : self.duration_period,
            'threshold' : self.threshold,
            'probability_threshold' : self.probability_threshold,
            'history_reward' : self.history_reward,
            'A_c' : self.A_c ,
            'b_c' : self.b_c,
            'p_c_t' : self.p_c_t,
            'context' : self.context,
            'other_context' : self.other_context
        }
        with open(file_name, 'wb') as f:
            pickle.dump(state, f)
        

    def load_model(self, model_num = -1):
        if model_num == -1: #无参默认加载最新模型
            file_name = f'train_dir/save_model{self.save_model_num - 1}.pkl'
        else:
            file_name = f'train_dir/save_model{model_num}.pkl'
        
        if not os.path.isfile(file_name):
            print(f"Error: File {file_name} does not exist.")
            return None
        print('--------------before load-------------------')
        print_model_state(self)

        with open(file_name, 'rb') as f:
            print('load model ',file_name)
            state = pickle.load(f)
        # update
        self.all_apps = state['all_apps']
        self.num_apps = state['num_apps']
        self.n_arms = state['n_arms']
        self.n_features = state['n_features']

        self.init_factor = state['init_factor']
        self.alpha = state['alpha']
        self.factor_alpha = state['factor_alpha']

        self.times = state['times']
        self.sampling_model = state['sampling_model']
        self.sample_result = state['sample_result']
        self.sample_config = state['sample_config']
        self.sample_times = state['sample_times']
        self.ratio = state['ratio']

        self.curr_best_config = state['curr_best_config']
        self.curr_best_reward = state['curr_best_reward']
        self.duration_period = state['duration_period']
        self.threshold = state['threshold']
        self.probability_threshold = state['probability_threshold']


        self.history_reward = state['history_reward']

        self.A_c = state['A_c']
        self.b_c = state['b_c']
        self.p_c_t = state['p_c_t']
        self.context = state['context']
        self.other_context = state['other_context']
        
        print('--------------after load-------------------')
        print_model_state(self)
    

        

if __name__ == '__main__':
    pass