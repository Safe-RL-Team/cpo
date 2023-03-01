import gym
import torch
import pickle
import numpy as np
from core.agent import Agent

env_name = 'LunarLander-v2'

env = gym.make(
    env_name,
    render_mode='human',
)
observation, info = env.reset(seed=42)

state_dim = env.observation_space.shape[0]
print(state_dim)
is_disc_action = len(env.action_space.shape) == 0
print(env.action_space.shape)
#running_state = ZFilter((state_dim,), clip=5)

#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/MountainCar_1/2023-01-13-exp-1-MountainCar-v0/intermediate_model/model_iter_50.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_1/2023-01-15-exp-1-LunarLander-v2/intermediate_model/model_iter_10.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_2/2023-01-15-exp-1-LunarLander-v2/intermediate_model/model_iter_500.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_vel/2023-01-15-exp-1-LunarLander-v2/intermediate_model/model_iter_330.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_vel_1/2023-01-15-exp-1-LunarLander-v2/intermediate_model/model_iter_1000.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_vel/2023-02-24-exp-1-CartPole-v1/model.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_own_vel/2023-02-27-exp-5-LunarLander-v2/model.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_own_vel/2023-02-27-exp-5-LunarLander-v2/intermediate_model/model_iter_1000.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_own_angle/2023-02-28-exp-2-LunarLander-v2/intermediate_model/model_iter_1000.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_own_vel/2023-03-01-exp-1-LunarLander-v2/model.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_own_pos/2023-03-01-exp-7-LunarLander-v2/intermediate_model/model_iter_100.p'
model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_own_vel/2023-03-01-exp-3-LunarLander-v2/intermediate_model/model_iter_200.p'
policy_net, _ = pickle.load(open(model_path, "rb"))
device = 'cpu'
policy_net.to(device)

"""create agent"""
#agent = Agent(env, policy_net, device, running_state=running_state, render=True)


for _ in range(1000):
   state_var = torch.tensor(observation, dtype = torch.float64).unsqueeze(0)
   print(np.sum(np.abs(observation[2:4])))
   #if not 0.15 > observation[1] > -0.15:
   #print(observation[4])
   #action = env.action_space.sample()
   action = policy_net.select_action(state_var)[0]
   #print(action, int(action))
   observation, reward, terminated, truncated, info = env.step(int(action.detach().numpy()))
   #env.render()

   if terminated or truncated:
      observation, info = env.reset()
env.close()
