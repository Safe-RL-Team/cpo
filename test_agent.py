import gym
import torch
import pickle
import numpy as np
from core.agent import Agent

def test(render=False):

    env_name = 'CartPole-v1'
    #env_name = 'LunarLander-v2'
    #env_name = 'BipedalWalker-v3'

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

    state_dim = env.observation_space.shape[0]
    print(state_dim)
    is_disc_action = len(env.action_space.shape) == 0
    print(env.action_space.shape)
    #running_state = ZFilter((state_dim,), clip=5)

    #CartPole_test, 18.3., max_kl=0.01, cg-its=20
    model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_test/2023-03-18-exp-5-CartPole-v1/model.p' #ils

    #Lunarlander, 18.3.
    #exp4 ls, max_kl=0.01
    #exp5 ils, max_kl=0.01, d_k=18
    model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_test/2023-03-18-exp-5-LunarLander-v2/model.p'
    # exp5 ils, max_kl=0.005

    # Lunarlander, 20.3.
    # exp1 ils, max_kl=0.01, d_k=25
    model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_test/2023-03-20-exp-1-LunarLander-v2/model.p'
    model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_test/2023-03-20-exp-2-LunarLander-v2/model.p'

    #CartPole CG test
    model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_CG/2023-03-21-exp-5-CartPole-v1/model.p'
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_CG_goleft/2023-03-21-exp-1-CartPole-v1/intermediate_model/model_iter_70.p'

    #CartPole go left
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_CG_goleft/2023-03-21-exp-2-CartPole-v1/model.p'
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_CG_goleft/2023-03-21-exp-2-CartPole-v1/intermediate_model/model_iter_490.p'
    model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_CG_goleft/2023-03-22-exp-11-CartPole-v1/best_training_model.p'
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_CG_goleft/2023-03-22-exp-11-CartPole-v1/intermediate_model/model_iter_100.p'

    #LunarLander CG
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_CG_angle/2023-03-21-exp-1-LunarLander-v2/model.p'
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_CG_angle/2023-03-21-exp-1-LunarLander-v2/intermediate_model/model_iter_100.p'

    #BipedalWalker CG
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/BipedalWalker_angular/2023-03-21-exp-1-BipedalWalker-v3/model.p'
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/BipedalWalker_angular/2023-03-21-exp-1-BipedalWalker-v3/intermediate_model/model_iter_100.p'

    #CartPole_test
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_test/2023-03-21-exp-2-CartPole-v1/model.p'
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_CG_pos/2023-03-22-exp-11-CartPole-v1/intermediate_model/model_iter_200.p'
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_CG_pos/2023-03-22-exp-11-CartPole-v1/model.p'

    #LunarLander, CG other Fvp
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_CG_vel/2023-03-22-exp-11-LunarLander-v2/model.p'
    #model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_CG_vel/2023-03-22-exp-12-LunarLander-v2/intermediate_model/model_iter_100.p'

    #LunarLander, kl
    model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_new_kl/2023-03-23-exp-2-LunarLander-v2/model.p'
    model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_left_manyIts/2023-03-26-exp-1-CartPole-v1/intermediate_model/model_iter_210.p'
    model_path = '/home/pauel/PycharmProjects/Sapana/PyTorch-CPO/assets/learned_models/CPO/CartPole_left_manyIts/2023-03-26-exp-1-CartPole-v1/intermediate_model/model_iter_210.p'

    try:
        policy_net, _, _ = pickle.load(open(model_path, "rb"))
    except:
        policy_net, _ = pickle.load(open(model_path, "rb"))
    device = 'cpu'
    policy_net.to(device)
    """create agent"""
    #agent = Agent(env, policy_net, device, running_state=running_state, render=True)


    speed = [[]]

    for i in range(20000):
       state_var = torch.tensor(observation, dtype = torch.float64).unsqueeze(0)
       #speed[-1].append(np.sum(np.abs(observation[2:4])))
       #if not 0.15 > observation[1] > -0.15:
       #print(observation[0])
       #action = env.action_space.sample()
       action = policy_net.select_action(state_var)[0]
       #print(round(action.detach().numpy()[0]))
       observation, reward, terminated, truncated, info = env.step(int(action.detach().numpy()[0])) #int(action.detach().numpy())
       #env.render()

       if truncated or terminated: #i%150==0:
          observation, info = env.reset()
          #print(len(speed[-1]))
          #speed.append([])

    env.close()

    #return speed

if __name__ == '__main__':
    test(render=True)