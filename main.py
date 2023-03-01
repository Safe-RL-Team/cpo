### Based on the implementation of Sapana Chadhaury
### Slightly adjusted

import gym
import os
import sys
import pickle
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.continuous_policy import Policy
from models.discrete_policy import DiscretePolicy
from algos.trpo import trpo_step
from cpo import cpo_step
from core.common import estimate_advantages_own, estimate_constraint_value
from core.agent import Agent
from utils import *
from utils.model_saving import save_info

### own scripts
from utils.constraint_functions import *

CUDA_LAUNCH_BLOCKING = 1

# summarizing using tensorboard
from torch.utils.tensorboard import SummaryWriter

# Returns the current local date
from datetime import date

today = date.today()
print("Today date is: ", today)

# Parse arguments
args = parse_all_arguments()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print('using gpu')
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
print(state_dim)
is_disc_action = len(env.action_space.shape) == 0
print(env.action_space.shape)
running_state = ZFilter((state_dim,), clip=5)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# env.seed(args.seed)

"""create all the paths to save learned models/data"""
save_info_obj = save_info(assets_dir(), args.exp_num, args.exp_name, args.env_name)  # model saving object
print(assets_dir())
save_info_obj.create_all_paths()  # create all paths
writer = SummaryWriter(os.path.join(assets_dir(), save_info_obj.saving_path, 'runs/'))  # tensorboard summary

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, render=args.render, num_threads=args.num_threads)

"""define constraint cost function"""


def constraint_cost(state, action):
    """Add your cosntraint cost function here"""
    if args.env_name == "LunarLander-v2":
        costs = LunarLander_vel(state, dtype, device, bound=2.5)
        # print('costs', torch.sum(costs).item())
    elif args.env_name == "CartPole-v1":
        costs = CartPole_vel(state, dtype, device)
    else:
        costs = tensor(0.01 * np.ones(state.shape[0]), dtype=dtype).to(device)
    # print('costs', costs, costs.shape)
    return costs


def update_params(batch, d_k=0):
    states = torch.from_numpy(np.stack(batch.state)[:args.max_batch_size]).to(dtype).to(device)  # [:args.batch_size]
    actions = torch.from_numpy(np.stack(batch.action)[:args.max_batch_size]).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)[:args.max_batch_size]).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)[:args.max_batch_size]).to(dtype).to(device)

    """get advantage estimation from the trajectories"""
    advantages = estimate_advantages_own(rewards, masks, args.gamma, args.tau, device)
    #print(type(advantages))

    if args.algo_name == "CPO":
        costs = constraint_cost(states, actions)
        cost_advantages = estimate_advantages_own(costs, masks, args.gamma, args.tau, device)
        constraint_value = estimate_constraint_value(costs, masks, args.gamma, device)
        constraint_value = constraint_value[0]

        # print('adv',cost_advantages, 'constr', constraint_value)

    """perform update"""
    if args.algo_name == "CPO":
        p_loss, cost_loss, constraint_value, r1, r2 = cpo_step(args.env_name, policy_net, states, actions, advantages,
                                             cost_advantages, constraint_value, d_k, args.max_kl, args.damping,
                                             args.l2_reg)
    elif args.algo_name == "TRPO":
        v_loss, p_loss = trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl,
                                   args.damping, args.l2_reg)
        cost_loss = 0

    return p_loss, cost_loss, constraint_value, r1, r2


def main_loop():
    # variables and lists for recording losses
    p_loss = 0
    cost_loss = 0
    constrain_value = 0
    p_loss_list = []
    cost_loss_list = []
    constrain_value_list = []

    # lists for dumping plotting data for agent
    rewards_std = []
    env_avg_reward = []
    num_of_steps = []
    num_of_episodes = []
    total_num_episodes = []
    total_num_steps = []
    residual1_list = []
    residual2_list = []
    time_list = []
    tne = 0  # cummulative number of episodes
    tns = 0  # cummulative number of steps

    # lists for dumping plotting data for mean agent
    eval_avg_reward = []
    eval_avg_reward_std = []

    if args.algo_name == "CPO":
        # define initial d_k
        d_k = args.max_constraint
        # define annealing factor
        if args.anneal == True:
            e_k = args.annealing_factor
        else:
            e_k = 0

            # for saving the best model
    if args.env_name in ["CartPole-v0", "CartPole-v1", "MountainCar-v0", 'LunarLander-v2']:
        best_std = 20
        best_avg_reward = -200
    else:
        best_std = 5
        best_avg_reward = 0

    for i_iter in range(args.max_iter_num):
        print(i_iter)
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size)

        t0 = time.time()
        if args.algo_name == "CPO":
            #print('d_k', d_k)
            p_loss, cost_loss, constrain_value, r1, r2 = update_params(batch, d_k)
        elif args.algo_name == "TRPO":
            v_loss, p_loss, cost_loss = update_params(batch)
        t1 = time.time()

        time_list.append(t1-t0)

        if args.algo_name == "CPO":
            d_k = d_k + d_k * e_k  # max constraint annealing

        # update lists for saving
        p_loss_list.append(p_loss)
        cost_loss_list.append(cost_loss)
        residual1_list.append(r1)
        residual2_list.append(r2)
        constrain_value_list.append(constrain_value)
        rewards_std.append(log['std_reward'])
        env_avg_reward.append(log['env_avg_reward'])
        num_of_steps.append(log['num_steps'])
        num_of_episodes.append(log['num_episodes'])
        tne = tne + log['num_episodes']
        tns = tns + log['num_steps']
        total_num_episodes.append(tne)
        total_num_steps.append(tns)

        # update tensorboard summaries
        #writer.add_scalars('losses', {'v_loss': v_loss}, i_iter)
        writer.add_scalars('losses', {'p_loss': p_loss}, i_iter)
        writer.add_scalars('losses', {'cost_loss': cost_loss}, i_iter)
        writer.add_scalars('data/rewards', {'env_avg_reward': log['env_avg_reward']}, i_iter)  # env_avg_reward
        writer.add_scalar('env_avg_reward', log['env_avg_reward'], i_iter)
        writer.add_scalar('num of episodes', log['num_episodes'], i_iter)
        writer.add_scalar('num of steps', log['num_steps'], i_iter)

        # evaluate the current policy
        running_state.fix = True  # Fix the running state
        agent.num_threads = 20
        if args.env_name == "CartPole-v0" or args.env_name == "CartPole-v1" or args.env_name == "MountainCar-v0":
            agent.mean_action = False
        else:
            agent.mean_action = True
        seed = np.random.randint(1, 1000)
        # agent.env.seed(seed)
        eval_reward_type = 4
        _, eval_log = agent.collect_samples(20000)
        running_state.fix = False
        agent.num_threads = args.num_threads
        agent.mean_action = False

        # update eval lists
        eval_avg_reward.append(eval_log['env_avg_reward'])
        eval_avg_reward_std.append(eval_log['std_reward'])

        # print learning data on screen
        #if i_iter % args.log_interval == 0:
            #print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_avg {:.2f}\tTest_R_avg {:.2f}\tTest_R_std {:.2f}'.format(
                #i_iter, log['sample_time'], t1 - t0, log['env_avg_reward'], eval_log['env_avg_reward'],
                #eval_log['std_reward']))

            # save the best model
        if eval_log['env_avg_reward'] >= best_avg_reward and eval_log['std_reward'] <= best_std:
            print('Saving new best model !!!!')
            to_device(torch.device('cpu'), policy_net)
            save_info_obj.save_models(policy_net, running_state)
            to_device(device, policy_net)
            best_avg_reward = eval_log['env_avg_reward']
            best_std = eval_log['std_reward']
            iter_for_best_avg_reward = i_iter + 1

        # save some intermediate models to sample trajectories from
        if args.save_intermediate_model > 0 and (i_iter + 1) % args.save_intermediate_model == 0:
            to_device(torch.device('cpu'), policy_net)
            save_info_obj.save_intermediate_models(policy_net, running_state, i_iter)
            to_device(device, policy_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()

    # dump expert_avg_reward, num_of_steps, num_of_episodes
    # save_info_obj.dump_lists(avg_reward, num_of_steps, num_of_episodes, total_num_episodes, total_num_steps, rewards_std, env_avg_reward, v_loss_list, p_loss_list, eval_avg_reward, eval_avg_reward_std)
    save_info_obj.dump_lists(num_of_steps, num_of_episodes, total_num_episodes, total_num_steps,
                             rewards_std, env_avg_reward, p_loss_list, eval_avg_reward,
                             eval_avg_reward_std)
    save_info_obj.save_info(eval_avg_reward, constrain_value_list, residual1=residual1_list, residual2=residual2_list, time=time_list)

    print('Best eval R:', best_avg_reward)
    return best_avg_reward, best_std, iter_for_best_avg_reward


main_loop()