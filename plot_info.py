import matplotlib.pyplot as plt
from numpy import genfromtxt
from test_agent_clean import test_LunarLander as test
import os
import csv

"""Here can be found various methods to plot the learning progress of an agent"""

file = 'assets/learned_models/CPO/CartPole/2023-03-04-exp-3-CartPole-v1/losses.csv' #d_k = 3

file1 = 'assets/learned_models/CPO/LunarLander_kl/2023-03-04-exp-1-LunarLander-v2/losses.csv' #d_k = 20, max_kl = 0.5*1e-2
file2 = 'assets/learned_models/CPO/LunarLander/2023-03-04-exp-2-LunarLander-v2/losses.csv' #d_k = 20, max_kl = 1e-2
file3 = 'assets/learned_models/CPO/LunarLander_kl/2023-03-05-exp-2-LunarLander-v2/losses.csv' #d_k = 20, max_kl = 2*1e-2
file4 = 'assets/learned_models/CPO/LunarLander_kl/2023-03-05-exp-3-LunarLander-v2/losses.csv' #d_k = 20, max_kl = 4*1e-2
file4 = 'assets/learned_models/CPO/LunarLander_kl/2023-03-05-exp-4-LunarLander-v2/losses.csv' #d_k = 20, max_kl = 0.1

file = 'assets/learned_models/CPO/CartPole_test/2023-03-17-exp-1-CartPole-v1/losses.csv'
file = 'assets/learned_models/CPO/LunarLander_test/2023-03-18-exp-2-LunarLander-v2/losses.csv'
file = 'assets/learned_models/CPO/LunarLander_test/2023-03-20-exp-1-LunarLander-v2/losses.csv'
file = 'assets/learned_models/CPO/CartPole_CG/2023-03-21-exp-4-CartPole-v1/losses.csv'
file = 'assets/learned_models/CPO/BipedalWalker_angular/2023-03-21-exp-1-BipedalWalker-v3/losses.csv'
file = 'assets/learned_models/CPO/LunarLander_CG_vel/2023-03-22-exp-1-LunarLander-v2/losses.csv'
file = 'assets/learned_models/CPO/LunarLander_CG_vel/2023-03-22-exp-11-LunarLander-v2/losses.csv'
file = 'assets/learned_models/CPO/CartPole_CG_pos/2023-03-22-exp-11-CartPole-v1/losses.csv'
file = 'assets/learned_models/CPO/LunarLander_manyIts/2023-03-26-exp-1-LunarLander-v2'

def plot_speed(render=False):

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

def plot_reward_constraint(file):

    df = genfromtxt(file, delimiter=',')
    print(df)

    fig, (ax1, ax2) = plt.subplots(2)

    _, n = df.shape
    ax1.plot(range(1, n + 1), df[0], label='reward')
    ax1.legend()
    ax2.plot(range(1, n + 1), df[1], label='constraint')
    ax2.plot(range(1, n + 1), n*[25], label='limit', color='red')
    ax2.legend()

    plt.show()

def plots(files):

    dfs = []

    for file in files:
        dfs.append(genfromtxt(file, delimiter=','))

    fig, (ax1, ax2) = plt.subplots(2)

    for df, kl  in zip(dfs, [0.5*1e-2, 2*1e-2, 0.1]):
        #_, n = df.shape
        n = 31
        ax1.plot(range(1, n + 1), df[0][:n], label=f'reward, {kl}')
        ax1.legend()
        ax2.plot(range(1, n + 1), df[1][:n], label=f'constraint, {kl}')
        if kl == 0.1:
            ax2.plot(range(1, n + 1), n*[20], label='limit', color='red')
        ax2.legend()

    plt.show()

def plot_mult(files):

    dfs = []
    kls = []
    for file in files:
        dfs.append(genfromtxt(file+'/losses.csv', delimiter=','))
        kls.append(genfromtxt(file+'/parameters.csv', delimiter=',')[2])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    for df,i in zip(reversed(dfs), reversed(kls)):
        _, n = df.shape
        ax1.plot(range(1, n + 1), df[7], label=f'{i} CG iterations')
        plt.setp(ax1, ylabel='reward')
        ax1.legend()
        ax2.plot(range(1, n + 1), df[1]) #, label=f'{i} CG iterations')
        plt.setp(ax2, ylabel='constraint')
        if i == kls[0]:
            ax2.plot(range(1, n + 1), n*[3], label='limit', color='black')
        ax2.legend()
        ax3.plot(range(1, n + 1), df[3], label=f'{i} CG iterations')
        plt.setp(ax3, ylabel='residual')
        #ax3.plot(range(1, n + 1), df[4], label=f'r2, {i} iterations')
        #ax3.legend()
        ax4.plot(range(1, n + 1), df[2] * df[3]/df[5], label=f'k(A)*r1/||g||, {i} CG iterations')
        plt.setp(ax4, ylabel='k(A)*r1/||g||')
        plt.setp(ax4, xlabel='iterations')
        #ax4.plot(range(1, n + 1), df[2] * df[4], label=f'k(A)*r2, {i} CG iterations')
        #ax4.legend()

    plt.show()

def plot_one(file):

    df = genfromtxt(file, delimiter=',')
    _, n = df.shape

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle('100 CG iterations')
    ax1.plot(range(1,n+1),df[0])
    plt.setp(ax1, ylabel='reward')
    ax1.legend()
    ax2.plot(range(1,n+1),df[1])
    plt.setp(ax2, ylabel='constraint')
    ax2.plot(range(1, n + 1), n * [3], label='limit', color='black')
    ax2.legend()
    ax3.plot(range(1,n+1), df[3], label='r1')
    ax3.plot(range(1,n+1), df[4], label='r2')
    plt.setp(ax3, ylabel='residual')
    ax3.legend()
    ax4.plot(range(1,n+1), df[2]*df[3]/df[5], label='k(A)*r1/||g||')
    ax4.plot(range(1,n+1), df[2]*df[4], label='k(A)*r2')
    plt.setp(ax4, ylabel='relative forward error')
    plt.setp(ax4, xlabel='iterations')
    ax4.legend()

    plt.show()

def plot_learning_prog(file):

    df = genfromtxt(file + '/losses.csv', delimiter=',')
    params = genfromtxt(file + '/parameters.csv', delimiter=',')
    _, n = df.shape

    fig, (ax1, ax2) = plt.subplots(2)
    #fig.suptitle(f'{int(params[3])} delta')
    ax1.plot(range(1, n + 1), df[4])
    plt.setp(ax1, ylabel='reward')
    ax2.plot(range(1, n + 1), df[1])
    plt.setp(ax2, ylabel='constraint')
    plt.setp(ax2, xlabel='iterations')
    ax2.plot(range(1, n + 1), n * [20], label='limit', color='black')
    ax2.legend()

    plt.show()

def plot_residual(file):

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


def plot_contraint(files):
    dfs = []
    for file in files:
        dfs.append(genfromtxt(file+'/losses.csv', delimiter=','))

    fig, (ax1) = plt.subplots(1)

    for df, i in zip(dfs, [10, 100]):
        _, n = df.shape
        ax1.plot(range(1, n + 1), df[1], label=f'{i} CG iterations')
        plt.setp(ax1, ylabel='constraint')
        plt.setp(ax1, xlabel='iterations')
        if i == 100:
            ax1.plot(range(1, n + 1), n * [20], label='limit', color='black')
        ax1.legend()

    plt.show()

if __name__ == '__main__':
    files=[]
    for i in range(0,5): #[[2,1],[2,2],[3,3],[3,4]]:
        files.append(f'assets/learned_models/CPO/CartPole_new_kl/2023-03-23-exp-{i}-CartPole-v1')
        #files.append(f'assets/learned_models/CPO/LunarLander_new_kl/2023-03-23-exp-{i}-LunarLander-v2')
        #files.append(f'assets/learned_models/CPO/CartPole_CG_pos/2023-03-2{i[0]}-exp-{i[1]}-CartPole-v1')
        #file = f'assets/learned_models/CPO/CartPole_test/2023-03-18-exp-{i}-CartPole-v1/losses.csv'
        #plot(file)
    #plot_mult(files)

    #plot_contraint([files[0], files[3]])

    #for file in files:
    #    plot_learning_prog(file)


    #plot_speed(render=False)
    plot_learning_prog(file)
    #plots([file1, file3, file4])
    #plot_one(file)