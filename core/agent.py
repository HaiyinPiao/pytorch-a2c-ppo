import torch.multiprocessing as mp
from utils.replay_memory import Memory
from utils.torch import *
from torch.autograd import Variable
import math
import time
import numpy as np

def index_to_one_hot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot

def collect_samples(pid, obs_shape_n, act_shape_n, queue, env, policy, custom_reward, mean_action,
                    tensor, render, running_state, update_rs, min_batch_size, g_itr):
    n_agents = len(policy)
    torch.randn(pid, )
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    # EPS_MAX = 0.9995
    # eps_val = EPS_MAX**float(g_itr)
    # if eps_val < 0.1:
    #     eps_val = 0.1

    # while num_steps < min_batch_size:
    while num_steps < min_batch_size:
        state = env.reset()
        # print(state)
        # if running_state is not None:
        #     state = running_state(state, update=update_rs)
        reward_episode = 0

        for t in range(10000):
            num_steps += 1
            action = []
            rewards = []
            state_var = Variable(tensor(state).unsqueeze(0), volatile=True)
            if mean_action:
                # never arrived
                action = policy(state_var)[0].data[0].numpy()
            else:
                for i in range(n_agents):
                    # action = policy[i].select_ma_action(state_var, n_agents)[0].numpy()
                    action.append(policy[i].select_action(state_var[:,i,:])[0].numpy()[0])
                # freeze 
                action[1] = 0
                # action[0] = 0
                # eps = np.random.randn(action.size)*eps_val
                # action = action + eps
                # np.clip(action, -1., 1.)
                # print(action)

            # action = int(action) if policy.is_disc_action else action.astype(np.float64)
            one_hot_actions = []
            for i in range(n_agents):
                one_hot_actions.append(index_to_one_hot(action[i], act_shape_n[i]))

            # print(one_hot_actions)
            next_state, reward, done, _ = env.step(one_hot_actions)

            # Added for shaped reward by haiyinpiao.
            # for punishing the bipedwalker from stucking in where it is originally.
            # if (next_state[2]<0.2):
            #     reward -=2
            # -------------------------
            # print(reward)
            reward_episode += np.mean(reward[0])
            # if running_state is not None:
            #     next_state = running_state(next_state, update=update_rs)

            # if custom_reward is not None:
            #     reward = custom_reward(state, action)
            #     total_c_reward += reward
            #     min_c_reward = min(min_c_reward, reward)
            #     max_c_reward = max(max_c_reward, reward)

            # mask = 0 if done[0] else 1
            mask = done

            memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
                # time.sleep(0.1)
            # done[3] indicates if the good agents caught
            if done[1] or num_steps >= min_batch_size:
                break
            # if done[0]:
            #     break

            state = next_state

        # log stats
        # num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    # print(pid,"collected!")

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, obs_shape_n, act_shape_n, env_factory, policy, custom_reward=None, mean_action=False, render=False,
                 tensor_type=torch.DoubleTensor, running_state=None, num_threads=1):
        self.env_factory = env_factory
        self.policy = policy
        self.n_agents = len(policy)
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.tensor = tensor_type
        self.num_threads = num_threads
        self.env_list = []
        for i in range(num_threads):
            self.env_list.append(self.env_factory(i))

        self.obs_shape_n = obs_shape_n
        self.act_shape_n = act_shape_n

        self.whole_critic_state_dim = 0
        self.whole_critic_action_dim = 0
        for i in range(self.n_agents):
            self.whole_critic_state_dim += self.obs_shape_n[i]
            self.whole_critic_action_dim += 1
            # self.whole_critic_action_dim += self.act_shape_n[i]

    def collect_samples(self, min_batch_size, g_itr):

        t_start = time.time()
        if use_gpu:
            for i in range(self.n_agents):
                self.policy[i].cpu()
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = mp.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, self.obs_shape_n, self.act_shape_n, queue, self.env_list[i + 1], self.policy, self.custom_reward, self.mean_action,
                           self.tensor, False, self.running_state, False, thread_batch_size, g_itr)
            workers.append(mp.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, self.obs_shape_n, self.act_shape_n, None, self.env_list[0], self.policy, self.custom_reward, self.mean_action,
                                      self.tensor, self.render, self.running_state, True, thread_batch_size, g_itr)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        if use_gpu:
            for i in range(self.n_agents):
                self.policy[i].cuda()
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
