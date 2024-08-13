from abc import ABC, abstractmethod
import numpy as np


class ScheduleFrame:
    def __init__(self):
        pass

    @abstractmethod
    def select_arm(self):
        """
        Make a decision, select a new resource allocation
        Returns:
            list<int>: new allocation for every task, e.g. [15, 8, 7]
        """
        pass

    @abstractmethod
    def update(self, reward, chosen_arm):
        """
        update the parameter matrix according to the reward
        Args:
            reward float: reward value according to the performance, bigger is greater
            chosen_arm dict<taskname -> allocation: current resource allocation

        Returns:

        """
        pass

    @abstractmethod
    def get_now_reward(self, performance, context_info=None):
        """
        update the context and calculate the reward according to the performance
        Args:
            performance list<float>: performacne of every task, may be hitrate, latency, and so on
            context_info list<list<float>>: context of current environment

        Returns:
            float: A reward value calculated based on performance metrics
        """
        pass


class ConfigManagement:
    def __init__(self):
        pass

    @abstractmethod
    def receive_config(self):
        """
        get information of tasks, current allocation, performance, context from client
        Returns:
            list: [
                [name of tasks]                             # ['userA', 'userB', 'userC']
                [allocation of resource A for every task]   # [10, 10, 10]
                [performance of every task]                 # [0.21, 0.54, 0.65]
                [feature1 of every task]                    # [feature1_A, feature1_B, feature1_C] e.g.
                [feature2 of every task]                    # [feature2_A, feature2_B, feature2_C] e.g.
                ......
                [featureN of every task]
            ]
        """
        pass

    @abstractmethod
    def send_config(self, new_config):
        """
        set new allocation in client
        Args:
            new_config (list): [
                [name of tasks],                                # ['userA', 'userB', 'userC']
                [new allocation of resource A for every task]   # [15, 8, 7]
            ]
        Returns:
        """
        pass


def latin_hypercube_sampling(n, d, m, M):
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
    for i in range(len(samples)):
        samples[i] = [int(ele) for ele in samples[i]]
        curr_sum = sum(samples[i])
        samples[i][-1] += M - curr_sum

    return samples


if __name__ == '__main__':
    # Parameters
    num_samples = 20  # Number of samples
    num_dimensions = 3  # Number of dimensions
    min_value = 0  # Minimum value of each element
    total_sum = 240  # Total sum of elements in each sample

    # Generate samples
    samples = latin_hypercube_sampling(num_samples, num_dimensions, min_value, total_sum)
    print(samples)

    pass
