import torch
import numpy as np

#Here is a collection of various constraint function we created and used during training

def R(a):

    b = torch.cat((a.unsqueeze(-1), torch.zeros_like(a).unsqueeze(-1)),1)
    c, _ = torch.max(b,1)

    return c

### LunarLander

def LunarLander_pos(state, dtype, device, bound=0.15):

    # state[:, 0] position on the x-axis of the aircraft
    # the aircraft is constrained to stay within [-bound, bound]
    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = (-bound > state[:, 0]) | (bound < state[:, 0])
    costs[idx] = 1

    return costs

def LunarLander_angle(state, dtype, device, bound=0.5):

    # state[:, 4] angle of the aircraft, ranges from -pi to pi
    # the aircraft is constrained to stay within an angle of [-bound, bound]
    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = torch.abs(state[:, 4]) > bound
    costs[idx] = 1

    return costs


def LunarLander_vel(state, dtype, device, bound=1.5):

    # state[:, 2] velocity in x direction, state[:, 3] velocity in y direction
    # the aircraft is constrained to stay below a velocity of bound
    # where here the velocity is measured as |state[:, 2]|+|state[:, 3]|
    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = torch.abs(state[:, 2]) + torch.abs(state[:, 3]) > bound
    costs[idx] = 1

    return costs

def LunarLander_vel_ReLU(state, dtype, device, bound=0.5):

    # state[:, 2] velocity in x direction, state[:, 3] velocity in y direction
    # raises the constraint cost linearly when certain threshhold is surpassed
    costs = R(torch.abs(state[:, 2]) + torch.abs(state[:, 3]) - bound)

    return costs

### CartPole

def CartPole_vel(state, dtype, device, bound=2):

    # state[:, 1] velocity of the pole, ranges from -inf to inf
    # the cart is constrained to stay below a velocity of bound
    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = torch.abs(state[:, 1]) > bound
    costs[idx] = 1

    return costs

def CartPole_pos(state, dtype, device, bound=2):

    # state[:, 0] position of the pole, ranges from -4.8 to 4.8
    # the cart is constrained to stay within the range [-bound, bound]
    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = torch.abs(state[:, 0]) > bound
    costs[idx] = 1

    return costs

def CartPole_go_left(state, dtype, device, bound=0):

    # state[:, 0] position of the pole, ranges from -4.8 to 4.8
    # the cart is constrained to stay within the range [-4.8, bound]
    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = torch.abs(state[:, 0]) > bound
    costs[idx] = 1

    return costs

#BipedalWalker

def BipedalWalker_avarage_angular_vel(state, dtype, device, bound=1):

    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = (torch.abs(state[:, 4]) + torch.abs(state[:, 6]) + torch.abs(state[:, 8]) + torch.abs(state[:, 10]))/4 > bound
    costs[idx] = 1

    return costs





