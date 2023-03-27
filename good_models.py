from test_agent_clean import *
from info_plot import plot_speed

#please specify the location of the project
project_loc = '/home/pauel/PycharmProjects/Sapana'

#LunarLander
#LunarLander, velocity, iterations 100, limit 20, delta 0.01, CG iterations 10, Batch-size 4000
LunarLander1 = 'PyTorch-CPO/assets/LL1.p'
#low kl (0.005), velocity, iterations 100, limit 20, delta 0.005, CG iterations 10, Batch-size 4000
LunarLander2 = 'PyTorch-CPO/assets/LL2.p'

#CartPole
#CartPole, stay mid, iterations 500, limit 3, delta 0.01, CG iterations 10, Batch-size 4000
CartPole1 = 'PyTorch-CPO/assets/CP1_model.p'
#CartPole, stay mid, iterations 500, limit 3, delta 0.01, CG iterations 10, Batch-size 4000
CartPole2 = 'PyTorch-CPO/assets/CP2_model.p'
#CartPole, stay mid, iterations 100, limit 6, delta 0.01, CG iterations 10, Batch-size 4000
CartPoleLeft = 'PyTorch-CPO/assets/CP_left.p'

def presentation():

    print('Here we present a small selection of some models we trained.')
    print('We mainly used the LunarLander and the CartPole environment.')
    print('First we have a look at an agent we trained on the CartPole environment.')
    print('We constrained him to stay in middle of the screen.')

    test_CartPole(model_path=project_loc+'/'+CartPole1, render=True)

    print('Here is another model, trained with the same parameters')

    test_CartPole(model_path=project_loc + '/' + CartPole2, render=True)

    print('We also tried other constraints, but these were not so successful.')
    print('We constrained the Cart to only move within the left third of the screen.')

    test_CartPole(model_path=project_loc + '/' + CartPoleLeft, render=True, runs=25)

    print('Now we come to the LunarLander environment.')
    print('In most of our experiments we constrained the aircraft to fly with low speed.')

    test_LunarLander(model_path=project_loc + '/' + LunarLander1, render=True)

    print('Here is another model, trained with the same parameters')
    print('This time we track how fast the aircraft has flown.')

    plot_speed(render=True)



presentation()
