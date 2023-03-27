import matplotlib.pyplot as plt
from numpy import genfromtxt
from test_agent_clean import test_LunarLander as test
import os
import csv

"""Here can be found various methods to plot the learning progress of an agent"""

# agent data is saved in losses.csv
# we save it in the following format
#1. test reward
#2. constrained value
#3. condition number of H
#4. residual of H^{-1}g
#5. residual of H^{-1}b
#6. 2-norm of g
#7. time per iteration
#8. training reward

def plot_speed(render=False):

    #Use this method to plot the trajectories of the speed of the aircraft of multiple runs

    speed = test(render=render, record_speed=True)
    #print(speed)
    n = len(speed[1])
    a = 0.2

    for s in speed[1:-2]:
        plt.plot(range(1, n+1), s, color='black', alpha=a)


    plt.plot(range(1, n + 1), speed[-2], label='speed', color='black', alpha=a)
    plt.plot(range(1, n + 1), n * [1.5], label='limit', color='red')
    plt.xlabel('speed')
    plt.ylabel('time steps')
    plt.legend()
    plt.show()


def plot_learning_prog(file):

    # plots the reward, the constraint value of a model

    df = genfromtxt(file + '/losses.csv', delimiter=',')
    params = genfromtxt(file + '/parameters.csv', delimiter=',')
    _, n = df.shape

    fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle(f'{int(params[3])} delta')
    ax1.plot(range(1, n + 1), df[4])
    plt.setp(ax1, ylabel='reward')
    ax2.plot(range(1, n + 1), df[1])
    plt.setp(ax2, ylabel='constraint')
    plt.setp(ax2, xlabel='iterations')
    ax2.plot(range(1, n + 1), n * [20], label='limit', color='black')
    ax2.legend()

    plt.show()


def plot_residual(file):

    #plot the residual and the normwise forward error which arises in the calculation of H^{-1}b, H^{-1}g

    df = genfromtxt(file + '/losses.csv', delimiter=',')
    params = genfromtxt(file + '/parameters.csv', delimiter=',')
    _, n = df.shape

    fig, (ax3, ax4) = plt.subplots(2)
    fig.suptitle(f'{int(params[3])} CG iterations')
    ax3.plot(range(1, n + 1), df[3], label='r1')
    ax3.plot(range(1, n + 1), df[4], label='r2')
    plt.setp(ax3, ylabel='residual')
    ax3.legend()
    ax4.plot(range(1, n + 1), df[2] * df[3] / df[5], label='k(H)*r1/||g||')
    ax4.plot(range(1, n + 1), df[2] * df[4], label='k(H)*r2')
    plt.setp(ax4, ylabel='relative forward error')
    plt.setp(ax4, xlabel='iterations')
    ax4.legend()

    plt.show()

def plot_mult(files):

    # plots all four plots of the latter two functions for multiple models

    dfs = []
    kls = []
    for file in files:
        dfs.append(genfromtxt(file+'/losses.csv', delimiter=','))
        kls.append(genfromtxt(file+'/parameters.csv', delimiter=',')[2])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    for df,i in zip(reversed(dfs), reversed(kls)):
        _, n = df.shape
        ax1.plot(range(1, n + 1), df[7], label=f'{i} delta')
        plt.setp(ax1, ylabel='reward')
        ax1.legend()
        ax2.plot(range(1, n + 1), df[1])
        plt.setp(ax2, ylabel='constraint')
        if i == kls[0]:
            ax2.plot(range(1, n + 1), n*[3], label='limit', color='black')
        ax2.legend()
        ax3.plot(range(1, n + 1), df[3], label=f'{i} CG iterations')
        plt.setp(ax3, ylabel='residual')
        ax4.plot(range(1, n + 1), df[2] * df[3]/df[5], label=f'k(A)*r1/||g||, {i} CG iterations')
        plt.setp(ax4, ylabel='k(A)*r1/||g||')
        plt.setp(ax4, xlabel='iterations')

    plt.show()

if __name__ == '__main__':

    #example plot

    files=[]
    for i in range(0,5):
        files.append(f'assets/learned_models/CPO/CartPole_new_kl/2023-03-23-exp-{i}-CartPole-v1')

    plot_mult(files)