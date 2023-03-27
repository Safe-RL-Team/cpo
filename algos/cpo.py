import numpy as np
import torch
import scipy
from utils import *
from own_utils.line_searches import line_search, intermediate_line_search
from own_utils.power_method import condition_number as cn
from utils.conjugate_gradients import conjugate_gradients


def cpo_step(env_name, policy_net, value_net, states, actions, returns, advantages, cost_advantages,
             constraint_value, d_k, max_kl, damping, l2_reg, cg_its, save_condition_number, use_fim=False):

    """update the value function"""
    #this is used in estimation of the advantage function in common.py
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()

    # pdb.set_trace()
    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).detach().cpu().numpy(),
                                                            maxiter=25)
    v_loss, _ = get_value_loss(get_flat_params_from(value_net).detach().cpu().numpy())
    set_flat_params_to(value_net, tensor(flat_params))

    """update policy"""
    with torch.no_grad():
        fixed_log_probs = policy_net.get_log_prob(states, actions)
    """define the loss function for Objective"""
    def get_loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_prob(states, actions)
            action_loss = -advantages * torch.exp(log_probs - fixed_log_probs)
            return action_loss.mean()


    """define the loss function for Constraint"""
    def get_cost_loss(volatile=False):
        with torch.set_grad_enabled(not volatile):
            log_probs = policy_net.get_log_prob(states, actions)
            cost_loss = cost_advantages * torch.exp(log_probs - fixed_log_probs)
            return cost_loss.mean()

    """directly compute Hessian*vector from KL"""
    def Fvp_direct(v):
        kl = policy_net.get_kl(states)
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

        return flat_grad_grad_kl + v * damping

    #define f_a(\lambda) and f_b(\lambda)
    def f_a_lambda(lamda):
        a = ((r**2)/s - q)/(2*lamda)
        b = lamda*((cc**2)/s - max_kl)/2
        c = - (r*cc)/s
        return a+b+c

    def f_b_lambda(lamda):
        a = -(q/lamda + lamda*max_kl)/2
        return a

    Fvp = Fvp_direct


    # Obtain objective gradient and step direction
    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach() #g
    # obtain (H^-1)*g by approximately solving Hx = g using CG-method
    stepdir, r1 = conjugate_gradients(Fvp, -loss_grad, cg_its) #(H^-1)*g


    #Obtain constraint gradient and step direction
    cost_loss = get_cost_loss()
    cost_grads = torch.autograd.grad(cost_loss, policy_net.parameters(), allow_unused=True)
    cost_loss_grad = torch.cat([grad.view(-1) for grad in cost_grads]).detach() #b
    cost_loss_grad = cost_loss_grad/torch.norm(cost_loss_grad) #normalize b
    # obtain (H^-1)*b by approximately solving Hx = b using CG-method
    cost_stepdir, r2 = conjugate_gradients(Fvp, -cost_loss_grad, cg_its) #(H^-1)*b
    cost_stepdir = cost_stepdir/torch.norm(cost_stepdir)  # normalize the cost-step-direction

    # Define q, r, s
    q = -loss_grad.dot(stepdir) #g^T * H^-1 * g
    r = loss_grad.dot(cost_stepdir) #g^T * H^-1 * b
    s = -cost_loss_grad.dot(cost_stepdir) #b^T * H^-1 * b

    # constaint value - limit of constraint
    d_k = tensor(d_k).to(constraint_value.dtype).to(constraint_value.device)
    cc = constraint_value - d_k # c would be positive for most part of the training

    # find optimal lambda_a and lambda_b
    # here A and B can be found by deriving the functions f_a_lambda and f_b_lambda respectively,
    # setting the derivative equal to zero and solving for lambda

    A = torch.sqrt(-(q - (r ** 2) / s) / (max_kl - (cc ** 2) / s))
    B = torch.sqrt(q / max_kl)

    # check wether A and B lie inside the respective intervals
    if cc>0:
        opt_lam_a = torch.max(r/cc,A)
        opt_lam_b = torch.max(0*A,torch.min(B,r/cc))
    else:
        opt_lam_b = torch.max(r/cc,B)
        opt_lam_a = torch.max(0*A,torch.min(A,r/cc))

    #find values of optimal lambdas by comparing their respective value and choosing the larger one
    opt_f_a = f_a_lambda(opt_lam_a)
    opt_f_b = f_b_lambda(opt_lam_b)

    if opt_f_a > opt_f_b:
        opt_lambda = opt_lam_a
    else:
        opt_lambda = opt_lam_b

    #find optimal nu
    nu = (opt_lambda*cc - r)/s
    if nu>0:
        opt_nu = nu
    else:
        opt_nu = 0

    """ find optimal step direction """
    # check for feasibility
    print('cc', cc.item())
    feasible = False
    if ((cc**2)/s - max_kl) > 0 and cc>0:
        print('INFEASIBLE !!!!')

        #use the recovery step
        opt_stepdir = torch.sqrt(2 * max_kl / s) * cost_stepdir
    else:
        feasible = True

        #use the found dual solutions to compute the primal solution
        opt_stepdir = (stepdir - opt_nu*cost_stepdir)/opt_lambda

    # perform line search as propesed in Achiam et al. to further enforce surrpgate constraint satisfaction
    prev_params = get_flat_params_from(policy_net)
    expected_improve = -loss_grad.dot(opt_stepdir)
    success, new_params = intermediate_line_search(policy_net, get_loss, prev_params, opt_stepdir, expected_improve, cc, cost_loss_grad, feasible)
    set_flat_params_to(policy_net, new_params)

    # save the conditionnumber of H for future evaluation
    #if save_condition_number:
    #    cnumber, t = cn(Fvp, opt_stepdir.shape[0])
    #    info = [cnumber,r1, r2, torch.norm(loss_grad).item(), t]
    #else:
    info = [r1, r2]

    return v_loss, loss, cost_loss, info
