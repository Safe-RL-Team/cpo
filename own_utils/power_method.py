import numpy as np
import torch
from utils.conjugate_gradients import conjugate_gradients as CG
import time

def condition_number(Avp_f, dim):

    #Calculate the 2-norm condition number K_2(A) = lambda_1 / lambda_n,
    #where we compute lambda_1, lambda_n by the power iteration and inverse power iteration method respectively
    #Note that in general K_2(A) =/= lambda_1 / lambda_n.
    #Here it is the case as the Hessian of the KL-divergence is symmetric
    #and hence especially normal

    t0 = time.time()
    q1 = power_iteration(Avp_f, dim)
    qn = inverse_power_iterarion(Avp_f, dim)

    def rayleigh(Avp_f, x):

        #returns the rayleigh quotient of A,x for x unit length w.r.t. the 2-norm

        return x.dot(Avp_f(x))

    return (rayleigh(Avp_f, q1) / rayleigh(Avp_f, qn)).item(), time.time() - t0

def power_iteration(Avp_f, dim, q=None, max_it=50):

    #power iteration method, returns an approximation of the largest eigenvalue in modulus

    it = 0

    if q==None:
        q = torch.ones(dim)
        q = q / dim**0.5

    while it < max_it:

        q = Avp_f(q)
        q = q / torch.linalg.norm(q)

        it += 1

    return q

def inverse_power_iterarion(Avp_f, dim, q=None, max_it=50):

    # inverse power iteration method, returns an approximation of the smallest eigenvalue in modulus

    it = 0

    if q==None:
        q = torch.ones(dim)
        q = q / dim**0.5

    while it < max_it:

        q, _ = CG(Avp_f, q, 10)
        q = q / torch.linalg.norm(q)

        it += 1

    return q