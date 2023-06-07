import torch
import numpy as np
from picture_to_dist import sample_elden_ring
from unif_transport import one_normalize
import matplotlib.pyplot as plt
from fit_kernel import clear_plt
from copy import deepcopy
from transport_kernel import l_scale, get_kernel

def regularity_test(s = 1):
    N = 10000
    n = 1000
    base_sample =  3 + torch.tensor(sample_elden_ring(N + 1)).T

    alpha_X_p = one_normalize((torch.linalg.norm(base_sample[:-1] , dim  = 1)**s).detach().cpu().numpy())

    beta_u = np.log(one_normalize(np.ones(N)))
    beta_X = np.log(alpha_X_p) - beta_u

    plt.plot(np.sort(alpha_X_p))
    plt.savefig('alpha_X_p.png')
    clear_plt()

    beta_Y = - deepcopy(beta_X)
    alpha_Y_p = one_normalize(np.exp(beta_Y + beta_u))
    plt.plot(np.sort(alpha_Y_p))
    plt.savefig('alpha_Y_p.png')
    clear_plt()


    idx = np.linspace(0, N, N).astype(int)

    sub_idx = np.random.choice(idx, size=n, replace=False)
    y_idx = np.random.choice(idx, size=n, replace=False, p=alpha_Y_p)

    beta_y = beta_Y[sub_idx]
    Y = base_sample[sub_idx, :]


    beta_x = beta_X[y_idx]
    Y_tilde = base_sample[y_idx, :]


    l = l_scale(Y)
    device = 'cpu'
    kernel_params = {'name': 'radial', 'l': l / 7, 'sigma': 1}
    kernel = get_kernel(kernel_params=kernel_params, device=device)
    nugget_matrix = 1e-7 * torch.eye(n, device=device)

    k_YY = kernel(Y, Y)
    k_YY_inv = torch.linalg.inv(k_YY + nugget_matrix).detach().cpu().numpy()

    k_YtYt = kernel(Y_tilde, Y_tilde)
    k_YtYt_inv = torch.linalg.inv(k_YtYt + nugget_matrix).detach().cpu().numpy()


    alpha_reg = beta_y @ k_YY_inv @ beta_y
    alpha_inv_reg = beta_x @ k_YtYt_inv @ beta_x

    return alpha_reg, alpha_inv_reg


def run():
    N = 100

    S_vals = np.linspace(-5, 5, num=N)
    reg_vals = []
    inv_reg_vals = []

    for s in S_vals:
        error = True
        while error:
            try:
                reg, inv_reg = regularity_test(s)
                reg_vals.append(reg)
                inv_reg_vals.append(inv_reg)
                error = False
            except:
                pass

    reg_vals = np.log(np.asarray(reg_vals))
    inv_reg_vals = np.log(np.asarray(inv_reg_vals))

    plt.scatter(reg_vals, inv_reg_vals)
    plt.xlabel('regularity')
    plt.ylabel('inverse regularity')
    plt.savefig('regularity v inverse regularity')


if __name__=='__main__':
    run()