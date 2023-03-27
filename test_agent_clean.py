import gym
import torch
import pickle
import numpy as np
from core.agent import Agent

#specify location of model in model_path to test a model

def test_CartPole(render=False, model_path=None, runs=10):

    env_name = 'CartPole-v1'

    if render:
        env = gym.make(
            env_name,
            render_mode='human',
        )
    else:
        env = gym.make(
            env_name
        )

    observation, info = env.reset(seed=42)

    if model_path == None:
        model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_new_kl/2023-03-23-exp-3-CartPole-v1/best_training_model.p'

    try:
        policy_net, _, _ = pickle.load(open(model_path, "rb"))
    except:
        policy_net, _ = pickle.load(open(model_path, "rb"))
    device = 'cpu'
    policy_net.to(device)
    """create agent"""

    run = 0
    for i in range(20000):
        state_var = torch.tensor(observation, dtype=torch.float64).unsqueeze(0)
        action = policy_net.select_action(state_var)[0]
        observation, reward, terminated, truncated, info = env.step(
            int(action.detach().numpy()[0]))  # int(action.detach().numpy())

        if truncated or terminated:
            observation, info = env.reset()
            run += 1
        if run >= runs:
            break

    env.close()

def test_LunarLander(render=False, model_path=None, runs = 10, record_speed=False):



    env_name = 'LunarLander-v2'

    if render:
        env = gym.make(
            env_name,
            render_mode='human',
        )
    else:
        env = gym.make(
            env_name
        )

    observation, info = env.reset(seed=42)

    if model_path == None:
        model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_manyIts/2023-03-26-exp-1-LunarLander-v2/best_training_model.p'

    try:
        policy_net, _, _ = pickle.load(open(model_path, "rb"))
    except:
        policy_net, _ = pickle.load(open(model_path, "rb"))
    device = 'cpu'
    policy_net.to(device)


    speed = [[]]
    run = 0
    for i in range(20000):
       state_var = torch.tensor(observation, dtype = torch.float64).unsqueeze(0)
       speed[-1].append(np.sum(np.abs(observation[2:4])))
       action = policy_net.select_action(state_var)[0]
       observation, reward, terminated, truncated, info = env.step(int(action.detach().numpy()[0])) #int(action.detach().numpy())

       if (i%175==0 and record_speed) or (not record_speed and (truncated or terminated)):
          observation, info = env.reset()
          speed.append([])
          run += 1

       if run >= runs:
           break

    env.close()

    return speed

if __name__ == '__main__':
    test_CartPole(render=True)
    #test_LunarLander(render=True)