import torch
import numpy as np
import os
from transport_kernel import get_kernel, l_scale
from unif_transport import unif_diffs, circle_diffs
from pydiffmap import diffusion_map as dm
from get_data import unif_circle, normal_theta_circle
import matplotlib.pyplot as plt
from copy import deepcopy


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True


def geo_diffs(X, Y = [], n_evecs=2, k=200, epsilon=.5, alpha=.5):
    N_x = max(list(X.shape))
    N_y = 0
    if len(Y):
        N_y = max(list(Y.shape))
        XY = np.concatenate((X,Y), axis = 1)
        Z = deepcopy(XY)
    else:
        Z = deepcopy(X)
    neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs=n_evecs, k=k, epsilon= epsilon, alpha= alpha,
                                          neighbor_params=neighbor_params)
    dmap = mydmap.fit_transform(Z.T).T
    W_diff = unif_diffs(dmap)[0].detach().cpu().numpy()
    if N_y:
        W_diff_XX = W_diff[:N_x, :N_x]
        W_diff_XY = W_diff[:N_x, N_x:]
        W_dif_YY = W_diff[N_x:, N_x:]
        return W_diff_XX ,W_diff_XY,W_dif_YY
    return W_diff


def run():
    N = 2000
    X = unif_circle(2500).T
    Y = normal_theta_circle(500)
    #Xt = deepcopy(X)
    #neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
    #mydmap = dm.DiffusionMap.from_sklearn(n_evecs=2, k=50, epsilon='bgh', alpha=1.0,
                                          #neighbor_params=neighbor_params)
    #dmap = mydmap.fit_transform(Xt.T).T

    #print(X.shape)


    W_XX ,W_XY, W_YY = geo_diffs(X, Y)

    WYY_theta = circle_diffs(torch.tensor(Y), torch.tensor(Y))[0].detach().cpu().numpy()
    WYY_unif = unif_diffs(Y,Y)[0].detach().cpu().numpy()

    W_YY[WYY_theta==0] = 1
    WYY_unif[WYY_unif == 0] = 1
    WYY_theta[WYY_theta == 0] = 1


    diff_ratios = np.log((W_YY/WYY_theta).flatten())
    sorted_ratios = np.sort(diff_ratios)
    plt.plot(sorted_ratios)
    plt.savefig('geo_diff_rationsYY.png')
    clear_plt()


    unif_diff_ratios = np.log(((WYY_unif/WYY_theta))).flatten()
    sorted_ratios = np.sort(unif_diff_ratios)
    plt.plot(sorted_ratios)
    plt.savefig('unif_diff_rationsYY.png')
    clear_plt()





if __name__=='__main__':
    run()
