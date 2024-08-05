import random
import numpy as np
from itertools import zip_longest
from ScheduleFrame import *


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
    for j in range(top_k):
        tmp = [[cache_top_k[0][j]] for _ in range(top_k ** (num_app - 1))]
        all_feasible_config.extend(gen_side(tmp, k=0))
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


class LinUCB(ScheduleFrame):
    def __init__(self, all_apps, n_cache, alpha, factor_alpha, n_features):
        super().__init__()
        self.all_apps = all_apps
        self.num_apps = len(all_apps)
        self.n_arms = n_cache + 1
        self.n_features = n_features
        self.alpha = alpha
        self.factor_alpha = factor_alpha
        self.times = 0

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

        return

    def select_arm(self):
        contexts = {}
        for app in self.all_apps:
            A = self.A_c[app]
            b = self.b_c[app]
            contexts[app] = np.hstack((self.context[app], self.other_context[app]))

            for i in range(self.n_arms):
                theta = np.linalg.inv(A[i]).dot(b[i])
                cntx = np.array(contexts[app])
                self.p_c_t[app][i] = theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(np.linalg.inv(A[i]).dot(cntx)))

        cache_action = beam_search(self.n_arms - 1, self.all_apps, self.p_c_t, self.times, end_condition=30)
        self.times += 1
        # cache_action is a dict
        cache_allocation = []
        for app in self.all_apps:
            cache_allocation.append(cache_action[app])
        self.alpha *= self.factor_alpha
        return cache_allocation

    def update(self, reward, chosen_arm):
        contexts = {}
        for app in self.all_apps:
            arm = chosen_arm[app]
            contexts[app] = np.hstack((self.context[app], self.other_context[app]))
            self.A_c[app][arm] += np.outer(np.array(contexts[app]), np.array(contexts[app]))
            self.b_c[app][arm] = np.add(self.b_c[app][arm].T, np.array(contexts[app]) * reward).reshape(self.n_features * 2, 1)

    def get_now_reward(self, performance, context_info=None):
        # update the context
        tmp = [list(row) for row in zip_longest(*context_info, fillvalue=None)]
        sum_context = np.zeros(self.n_features)
        for i, app in enumerate(self.all_apps):
            self.context[app] = tmp[i]
            sum_context += np.array(self.context[app])
        for app in self.all_apps:
            self.other_context[app] = list((sum_context - np.array(self.context[app])) / (len(self.all_apps) - 1))
        # calculate the reward
        th_reward = sum(float(x) for x in performance) / len(performance)
        return th_reward


if __name__ == '__main__':
    pass
