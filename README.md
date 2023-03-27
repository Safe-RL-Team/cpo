# PyTorch implementation of Constrained Policy Optimization (CPO)
This code is from Sapana Chadhaury https://github.com/SapanaChaudhary/PyTorch-CPO .
We did not make any substantial changes to the code.
Most of the Code found in own_utils was written by ourselves.
There can be the constraint functions we used and the methods we used for our experiments regarding the number of CG-iterations.
The CPO as described in Achiam et al. algorithm is implemented in algos/cpo.py .
Sadly we were not able to make the code fully functional, as we ran into many difficulties.
A presentation of some nice models we trained can be viewed by running good_models.py.
Methods for plotting agent data can be found in info_plot.py .
Methods for viewing the Agent in the environment can be found in test_agent_clean.py . 

## Pre-requisites
- [PyTorch](https://pytorch.org/get-started/previous-versions/#v120) (We used a new version of PyTorch but the original recommendation is PyTorch 1.2.0.) 
- OpenAI [Gym](https://github.com/openai/gym).


## Features 
Learning progress is saved in losses.csv for every model, a more detailed explanation of the format can be found in info_plot.py

### Usage
* python algos/main.py --env-name CartPole-v1 --algo-name=CPO --exp-num=1 --exp-name=CPO/CartPole --save-intermediate-model=10 --gpu-index=0 --max-iter=500

one can further also specify the number of CG iterations using --cg-iter


