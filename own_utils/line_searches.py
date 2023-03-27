from utils import *

#Original version from Sapana
def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x

#Version of Sapana with check for surrogate constraint satisfaction
def intermediate_line_search(model, f, x, fullstep, expected_improve_full, cc, cost_loss_grad, feasible, max_backtracks=10, accept_ratio=0.1, tol=0.1):
    fval = f(True).item()
    v = cost_loss_grad.dot(fullstep)

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        #if feasible check surrogate constraint satisfaction
        if ratio > accept_ratio and (cc + stepfrac*v <= tol or not feasible):
            return True, x_new
    return False, x

#Here if not feasible the opposite of a backtracking linesearch is done, something like fronttracking.
#As we often encountered the problem that for large deltas (max_kl) the recovery step lead to a heavy decrease in the reward,
#and basically mitigated all the progress made in the previous steps.
#We thought the reason for this might just be that the step chosen is simply to large.
#Unfortunately this method did not work as hoped.

def own_line_search(model, f, x, fullstep, expected_improve_full, cc, cost_loss_grad, feasible, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()
    v = cost_loss_grad.dot(fullstep)

    if feasible:
        stepfracs = [.5**x for x in range(max_backtracks)]
    else:
        stepfracs = [.5**x for x in [20, 15, 10, 8,4,3,2]]

    for stepfrac in stepfracs:
        if feasible:
            x_new = x + stepfrac * fullstep
            set_flat_params_to(model, x_new)
            fval_new = f(True).item()
            actual_improve = fval - fval_new
            expected_improve = expected_improve_full * stepfrac
            ratio = actual_improve / expected_improve

            #check surrogate constraint satisfaction
            if ratio > accept_ratio and cc + stepfrac*v <= 0:
                return True, x_new
        else:
            #check if surrogate constraint satisfaction might be already satisfied
            if cc + stepfrac*v <= 0:
                print('stepfrac:', stepfrac)
                return True, x + stepfrac * fullstep
    if feasible:
        return False, x
    else:
        #if surrogate constraint satisfaction was not achieved for a stepfrac<1 use the fullstep
        print('stepfrac:', 1)
        return True, x + fullstep