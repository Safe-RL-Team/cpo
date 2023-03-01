# PyTorch implementation of Constrained Policy Optimization (CPO)
This repository is based on the code of Sapana Chadhaury (https://github.com/SapanaChaudhary/PyTorch-CPO). 

## Pre-requisites
- [PyTorch](https://pytorch.org/get-started/previous-versions/#v120) (The code is tested on PyTorch 1.2.0.) 
- OpenAI [Gym](https://github.com/openai/gym).

### Usage
* python main.py --env-name CartPole-v1 --algo-name=CPO --exp-num=1 --exp-name=CPO/CartPole --save-intermediate-model=10 --gpu-index=0 --max-iter=500


