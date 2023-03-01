import torch
import numpy as np

def R(a):

    b = torch.cat((a.unsqueeze(-1), torch.zeros_like(a).unsqueeze(-1)),1)
    c, _ = torch.max(b,1)

    return c

def LunarLander_pos(state, dtype, device, bound=0.15):

    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = (-bound > state[:, 0]) | (bound < state[:, 0])
    costs[idx] = 1

    return costs

def LunarLander_angle(state, dtype, device, bound=0.25):

    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = torch.abs(state[:, 4]) > bound
    costs[idx] = 1

    return costs


def LunarLander_vel(state, dtype, device, bound=0.5):

    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = torch.abs(state[:, 2]) + torch.abs(state[:, 3]) > bound
    costs[idx] = 1

    return costs

def LunarLander_vel_ReLU(state, dtype, device, bound=0.5):

    costs = R(torch.abs(state[:, 2]) + torch.abs(state[:, 3]) - bound)

    return costs

def CartPole_vel(state, dtype, device, bound=1):

    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = torch.abs(state[:, 1]) > bound
    costs[idx] = 1

    return costs



