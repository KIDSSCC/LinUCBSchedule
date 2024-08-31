import socket
import time
import subprocess
import sys
from ScheduleFrame import ConfigManagement, ConfigPackage


cgroup_path = '/sys/fs/cgroup/blkio/'
host = '127.0.0.1'
port = 54000
disk_bandwidth = 512 * 1024
passwd = 'k15648611412'


def get_pid(task_name):
    all_pids = []
    for name in task_name:
        proc = subprocess.run(['pidof', name], shell=False, text=True, capture_output=True)
        pid = proc.stdout.strip().split()
        assert len(pid) != 0, '{} not found'.format(name)
        assert len(pid) == 1, '{} have more than one proc'.format(name)
        all_pids.append(pid[0])
    return all_pids

def get_cpu_allocation(pids):
    cpu_allocation = []
    for pid in pids:
        try:
            result = subprocess.run(['taskset', '-p', str(pid)], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to get CPU affinity for PID {pid}: {result.stderr.strip()}")
            output = result.stdout.strip()
            affinity_mask_hex = output.split()[-1]
            affinity_mask_bin = bin(int(affinity_mask_hex, 16))
            cpu_count = affinity_mask_bin.count('1')
            cpu_allocation.append(cpu_count)
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit()

    return cpu_allocation

def get_diskbandwidth_allocation(pids):
    pid_allocation = []
    for pid in pids:
        group_name = 'group_' + str(pid)
        check_command = 'sudo -S cgget -g blkio:' + group_name
        check_res = subprocess.run(check_command, input=passwd, shell=True, text=True, capture_output=True)
        if 'cannot' in check_res.stderr:
            print('non exist')
            pid_allocation.append(None)
        else:
            get_command = 'sudo -S cgget -r blkio.throttle.read_bps_device ' + group_name
            get_res = subprocess.run(get_command, input=passwd, shell=True, text=True, capture_output=True)
            read_diskbandwidth = get_res.stdout.split('\n')[1].split(' ')[-1]
            pid_allocation.append(int(read_diskbandwidth)/disk_bandwidth)

    return pid_allocation

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
    command = 'ls -d ' + cgroup_path + 'group*/'
    # print(command)
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    stdout = result.stdout
    if 'cannot' in stdout or "" == stdout:
        # no groups
        # print('no groups need to clear')
        return
    all_groups = stdout.strip().split('\n')
    all_groups = [line.replace(cgroup_path, "")[:-1] for line in all_groups]
    
    for group in all_groups:
        delete_command = 'sudo -S cgdelete -r blkio:' + group
        # print(delete_command)
        subprocess.run(delete_command, input=passwd, shell=True, text=True, capture_output=True)
    # num = len(all_groups)
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
        check_command = 'sudo -S cgget -g blkio:' + group_name
        # print(check_command)
        check_res = subprocess.run(check_command, input=passwd, shell=True, text=True, capture_output=True)
        if 'cannot' in check_res.stderr:
            # group non-exist,need to create new group
            print('{} non-exist'.format(group_name))
            # create new group
            create_command = 'sudo -S cgcreate -g blkio:' + group_name
            # print(create_command)
            subprocess.run(create_command, input=passwd, shell=True, text=True, capture_output=True)
            # add proc to group
            classify_command = 'sudo -S cgclassify -g blkio:' + group_name + ' ' + str(procs[i])
            # print(classify_command)
            subprocess.run(classify_command, input=passwd, shell=True, text=True, capture_output=True)
        # adjust the weigh
        adjust_command = 'sudo -S cgset -r blkio.throttle.read_bps_device="8:16 ' + \
                         str(bandwidths[i] * disk_bandwidth) + \
                         '" ' + group_name
        # print(adjust_command)
        subprocess.run(adjust_command, input=passwd, shell=True, text=True, capture_output=True)


class ProtoSystemManagement(ConfigManagement):
    def __init__(self) -> None:
        super().__init__()

    def receive_config(self):
        curr_config = ConfigPackage()
        # pool name and cache allocation
        pool_and_size = get_pool_stats()
        pool_name = list(pool_and_size.keys())
        pids = get_pid(pool_name)

        cache_allocation = list(pool_and_size.values())
        cpu_allocation = get_cpu_allocation(pids)

        curr_config.task_id = pool_name
        curr_config.resource_allocation.append(cache_allocation)
        curr_config.resource_allocation.append(cpu_allocation)
        
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
        curr_config.performance = performance
        # context
        n_features = 10
        for _ in range(n_features):
            curr_config.context.append([1.0] * len(pool_name))
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
        curr_config = ConfigPackage()
        curr_config.task_id = [u.name for u in self.all_user]
        curr_config.resource_allocation = [u.resources for u in self.all_user]
        curr_config.performance = [u.user_func(u.resources) for u in self.all_user]
        # context info
        for i in range(self.n_features):
            curr_config.context.append([1.0] * len(self.all_user))

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
    # clear_groups()
    task_name = ['leveldb_uniform_100M', 'mysql_uniform_100M']
    pids = get_pid(task_name)
    print(pids)
    get_res = get_diskbandwidth_allocation(pids)
    print('before:', get_res)
    set_bandwidth(pids, [2, 4])
    get_res = get_diskbandwidth_allocation(pids)
    print('after: ', get_res)
    
