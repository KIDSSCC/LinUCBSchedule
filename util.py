import socket
import time
import subprocess
from ScheduleFrame import ConfigManagement


cgroup_path = '/sys/fs/cgroup/blkio/'
host = '127.0.0.1'
port = 54000


def get_last_line(file_name):
    """
    Open file and read the last line

    Args:
        file_name (str): file name

    Returns:
        str: last line of the file, '' if the file not found
    """
    try:
        with open(file_name, 'rb') as file:
            file.seek(-2,2)
            while file.read(1) != b'\n':
                file.seek(-2, 1)
            return file.readline().decode().strip()
    except FileNotFoundError:
        return ''


def get_pool_stats():
    """
    Communicating with cache server through socket, get current cache pool name and cache allocation

    Args:

    Returns:
        map<poolName -> poolSize>: all pool in the cache and their pool size
    """
    # link the target program
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    message = "G:"
    # print(message)
    sock.sendall(message.encode())

    # wait the response
    response = sock.recv(1024).decode()
    # print(response)
    sock.close()
    deserialized_map = {}
    pairs = response.split(';')[:-1]
    for pair in pairs:
        key,value = pair.split(':')
        deserialized_map[key] = int(value)
    
    return deserialized_map


def clear_groups():
    """
    delete existing blkio groups
    Args:

    Returns:
    """
    command = 'ls -d ' + cgroup_path + '*/'
    print(command)
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    stdout = result.stdout
    if 'cannot' in stdout or "" == stdout:
        # no groups
        # print('no groups need to clear')
        return
    all_groups = stdout.strip().split('\n')
    all_groups = [line.replace(cgroup_path, "")[:-1] for line in all_groups]
    num = len(all_groups)
    for group in all_groups:
        delete_command = 'cgdelete -r blkio:' + group
        print(delete_command)
        subprocess.run(delete_command, shell=True, text=True, capture_output=False)
    # print('clear {} groups'.format(num))


def set_cache_size(workloads, cache_size):
    """
    Communicating with cache server through socket, adjust pool size to the new size

    Args:
        workloads (list<str>): pool name of all workloads
        cache_size (list<int>): new size of each pool

    Returns:
    """
    # link the server program
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    curr_config = [workloads, cache_size]
    serialized_data = '\n'.join([' '.join(map(str, row)) for row in curr_config])
    serialized_data = 'S:' + serialized_data
    # print(serialized_data)

    # send to server
    sock.sendall(serialized_data.encode())
    sock.close()


def set_bandwidth(procs, bandwidths):
    """
    Communicating with OS cgroup blkio, adjust bandwidth
    Args:
        procs (list<str>): pid of all workloads
        bandwidths (list<int>): new bandwidth of every workload

    Returns:

    """
    for i in range(len(procs)):
        group_name = 'group_' + str(procs[i])
        # check the group exist
        check_command = 'cgget -g blkio:' + group_name
        # print(check_command)
        check_res = subprocess.run(check_command, shell=True, text=True, capture_output=True)
        if 'cannot' in check_res.stderr:
            # group non-exist,need to create new group
            print('{} non-exist'.format(group_name))
            # create new group
            create_command = 'cgcreate -g blkio:' + group_name
            # print(create_command)
            subprocess.run(create_command, shell=True, text=True, capture_output=False)
            # add proc to group
            classify_command = 'cgclassify -g blkio:' + group_name + ' ' + str(procs[i])
            # print(classify_command)
            subprocess.run(classify_command, shell=True, text=True, capture_output=False)
        # adjust the weigh
        adjust_command = 'cgset -r blkio.throttle.read_bps_device="8:16 ' + \
                         str(bandwidths[i] * 102400) + \
                         '" ' + group_name
        # print(adjust_command)
        subprocess.run(adjust_command, shell=True, text=True, capture_output=False)


class ProtoSystemManagement(ConfigManagement):
    def __init__(self) -> None:
        super().__init__()

    def receive_config(self):
        curr_config = []
        # pool name and cache allocation
        pool_and_size = get_pool_stats()
        pool_name = list(pool_and_size.keys())
        pool_size = list(pool_and_size.values())
        curr_config.append(pool_name)
        curr_config.append(pool_size)
        # performance
        sub_item_log = ['/home/md/SHMCachelib/bin/0809/' + name + '_subItem.log' for name in pool_name]
        performance = []
        for log in sub_item_log:
            last_line = None
            while last_line is None or last_line == '':
                if last_line is None:
                    last_line = get_last_line(log)
                else:
                    time.sleep(10)
                    last_line = get_last_line(log)
            performance.append(last_line)
        curr_config.append(performance)
        # context
        n_features = 10
        for _ in range(n_features):
            curr_config.append([1.0] * len(pool_name))
        return curr_config

    def send_config(self, new_config):
        set_cache_size(new_config[0], new_config[1])


def uniform(resources):
    return min(resources, 120) * 0.015


def sequential(resources):
    return 0


def hotspot(resources):
    return min(resources, 30) * 0.035 + min(max(0, resources - 30), 60) * 0.01


class UserModel:
    def __init__(self, name, resources=0, user_func=None):
        self.name = name
        self.resources = resources
        self.user_func = user_func


class SimulationManagement(ConfigManagement):
    def __init__(self):
        super().__init__()
        self.total_resource = 113
        self.all_user = [
            UserModel('A', 0, sequential),
            UserModel('B', 0, uniform),
            UserModel('C', 0, hotspot),
            UserModel('D', 0, sequential),
            UserModel('E', 0, hotspot),
        ]
        for u in self.all_user:
            u.resources = self.total_resource // len(self.all_user)

        self.n_features = 10
        self.workload_change = -1
        self.counter = 0

    def receive_config(self):
        curr_config = [
            [u.name for u in self.all_user],
            [u.resources for u in self.all_user],
            [u.user_func(u.resources) for u in self.all_user]
        ]

        # context info
        for i in range(self.n_features):
            curr_config.append([1.0] * len(self.all_user))

        self.counter = self.counter + 1
        if self.workload_change > 0 and self.counter % self.workload_change == 0:
            print('---------------------------workload change -----------------------------')
            self.all_user[0].user_func, self.all_user[1].user_func = self.all_user[1].user_func, self.all_user[0].user_func
            self.all_user[1].user_func, self.all_user[2].user_func = self.all_user[2].user_func, self.all_user[1].user_func
        return curr_config

    def send_config(self, new_config):
        for i in range(len(self.all_user)):
            self.all_user[i].resources = new_config[1][i]


if __name__ == '__main__':
    pass
