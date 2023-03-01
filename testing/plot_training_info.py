import matplotlib.pyplot as plt
from numpy import genfromtxt
import csv

#file = 'assets/learned_models/CPO/LunarLander_own_angle/2023-03-01-exp-1-LunarLander-v2/losses.csv'
#file = 'assets/learned_models/CPO/LunarLander_own_vel/2023-03-01-exp-2-LunarLander-v2/losses.csv'
#file = 'assets/learned_models/CPO/LunarLander_own_pos/2023-03-01-exp-3-LunarLander-v2/losses.csv'
file = '../assets/learned_models/CPO/LunarLander_own_vel/2023-03-01-exp-3-LunarLander-v2/losses.csv'

df = genfromtxt(file, delimiter=',')

_,n = df.shape
#plt.plot(range(1,n+1),df[0], label='reward')
#plt.plot(range(1,n+1),df[1], label='constrained')

print(df[1])

fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')
ax1.plot(range(1,n+1),df[0], label='reward')
ax2.plot(range(1,n+1),df[1], label='constrained')
ax3.plot(range(1,n+1), df[2], label='r1')
ax3.plot(range(1,n+1), df[3], label='r2')
#plt.xlabel('iterations')
#plt.ylabel('reward/ constrained')
#plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()