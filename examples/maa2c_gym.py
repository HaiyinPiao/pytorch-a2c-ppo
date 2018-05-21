import argparse
import gym
import os
import sys
import pickle
import time
import torch as th
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/niao2/haiyinpiao/code_repo/pytorch-a2c-ppo/core')

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from torch.autograd import Variable
from core.a2c import a2c_step
from core.common import estimate_advantages
from core.agent import Agent

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')
# Tensor = FloatTensor
# torch.set_default_tensor_type('torch.FloatTensor')

parser = argparse.ArgumentParser(description='PyTorch A2C example')
parser.add_argument('--env-name', default="Hopper-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=0, metavar='G',
                    help='log std for the policy (default: 0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per A2C update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=10, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--num-adversaries', type=int, default=0, metavar='N',
                    help="num-adversaries (default: 0)")
args = parser.parse_args()

use_gpu = True
args.env_name = 'simple'
# args.model_path = '../assets/learned_models/Cartpole_a2c.p'

def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def env_factory(thread_id):
    # Create environment
    env = make_env(args.env_name)
    env.seed(args.seed + thread_id)
    return env

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

env_dummy = env_factory(0)

num_adversaries = min(env_dummy.n, args.num_adversaries)

obs_shape_n = [env_dummy.observation_space[i].shape[0] for i in range(env_dummy.n)]
act_shape_n = [env_dummy.action_space[i].n for i in range(env_dummy.n)]

state_dim = obs_shape_n[0]
action_dim = act_shape_n[0]

# is_disc_action = len(env_dummy.action_space[0]) == 0
is_disc_action = True

# ActionTensor = LongTensor if is_disc_action else DoubleTensor
ActionTensor = LongTensor if is_disc_action else FloatTensor

running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""define actor and critic"""
policy_net = []
value_net = []
if args.model_path is None:
    if is_disc_action:
        for i in range(env_dummy.n):
            policy_net.append(DiscretePolicy(obs_shape_n[i], act_shape_n[i]))
            # print(policy_net[i])
    else:
        policy_net = Policy(obs_shape_n[i], env_dummy.action_space.shape[0], log_std=args.log_std)
    # value_net = Value(state_dim)
    for i in range(env_dummy.n):
        value_net.append(Value(obs_shape_n[i]))
        # print(value_net[i])
else:
    # TODO
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
    # policy_net = [env_dummy.observation_space[i].shape[0] for i in range(env_dummy.n)]
if use_gpu:
    # policy_net = policy_net.cuda()
    # value_net = value_net.cuda()
    for i in range(env_dummy.n):
        policy_net[i].cuda()
        value_net[i].cuda()

optimizer_policy = []
optimizer_value = []
for i in range(env_dummy.n):
    optimizer_policy.append(torch.optim.Adam(policy_net[i].parameters(), lr=4e-4))
    optimizer_value.append(torch.optim.Adam(value_net[i].parameters(), lr=8e-4))

del env_dummy

"""create agent"""
agent = Agent(obs_shape_n, act_shape_n, env_factory, policy_net, running_state=running_state, render=args.render, num_threads=args.num_threads)

def to_tensor_var(x, use_cuda=True, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(DoubleTensor(x))
 

def update_params(batch):
    for i in range(len(policy_net)):
        policy_net[i].train()
        value_net[i].train()

    # states = torch.from_numpy(np.stack(batch.state))
    # actions = torch.from_numpy(np.stack(batch.action))
    # rewards = torch.from_numpy(np.stack(batch.reward))
    # masks = torch.from_numpy(np.stack(batch.mask).astype(np.float64))
    states = to_tensor_var(batch.state,True,"double").view(-1, agent.n_agents, agent.obs_shape_n[0])
    actions = to_tensor_var(batch.action,True,"double").view(-1, agent.n_agents, agent.act_shape_n[0])
    rewards = to_tensor_var(batch.reward,True,"double").view(-1, agent.n_agents, 1)
    masks = to_tensor_var(batch.mask,True,"double").view(-1, agent.n_agents, 1)

    whole_states_var = states.view(-1, agent.whole_critic_state_dim)
    whole_actions_var = actions.view(-1, 1)

    # print( whole_states_var, whole_actions_var )



    if use_gpu:
        states, actions, rewards, masks = states.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()
        whole_states_var, whole_actions_var = whole_states_var.cuda(), whole_actions_var.cuda()
    # values = value_net(Variable(whole_states_var, volatile=True)).data
    values = []
    for i in range(len(value_net)):
        # values.append(value_net[i](th.Tensor(whole_states_var)).data)
        # input = Variable(whole_states_var, volatile=True)
        values.append(value_net[i](whole_states_var))

    print(values)

    # """get advantage estimation from the trajectories"""
    # advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, use_gpu)

    # """perform TRPO update"""
    # a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, args.l2_reg)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, i_iter)
        t0 = time.time()
        update_params(batch)
        # for i in range(10):
        #     update_params(batch)
        #     # print("update", i_iter, "iteration", i, "updating-step")
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        # if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
        #     if use_gpu:
        #         policy_net.cpu(), value_net.cpu()
        #     pickle.dump((policy_net, value_net, running_state),
        #                 open(os.path.join(assets_dir(), 'learned_models/{}_a2c.p'.format(args.env_name)), 'wb'))
        #     if use_gpu:
        #         policy_net.cuda(), value_net.cuda()


main_loop()
