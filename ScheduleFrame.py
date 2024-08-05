from abc import ABC, abstractmethod


class ScheduleFrame:
    def __init__(self):
        pass

    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod
    def update(self, reward, chosen_arm):
        pass

    @abstractmethod
    def get_now_reward(self, performance, context_info=None):
        pass


if __name__ == '__main__':
    pass
