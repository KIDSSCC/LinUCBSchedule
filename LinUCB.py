import numpy as np

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
    print('reward is: '+ str(th_reward))
    return context, th_reward