import torch
import numpy as np
import os
from transport_kernel import get_kernel, l_scale
from unif_transport import unif_diffs, circle_diffs, normalize_rows, get_res_dict, resample
#from pydiffmap import diffusion_map as dm
from get_data import unif_circle, normal_theta_circle
import matplotlib.pyplot as plt
from copy import deepcopy

def dict_to_np(dict):
    for key,val in dict.items():
        try:
            dict[key] = val.detach().cpu().numpy()
        except BaseException:
            pass
    return dict

def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True


def topk_by_sort(input, k, axis=None, ascending=True):
    if not ascending:
        input *= -1
    ind = np.argsort(input, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind, axis=axis)
    return ind, val


def eta_knn(W, args):
    eta_W = np.zeros(W.shape)
    k = args['k']
    for i,row in enumerate(W):
        row_neighbors = topk_by_sort(row, k,  ascending=True)[0]
        eta_W[i,row_neighbors] = 1
    return eta_W


def get_W_geo(W_t0, W_knn, max_d = 50):
    W_geo = deepcopy(W_knn)
    W_t = deepcopy(W_t0)

    i = 0
    while not np.min(W_t) and i < max_d:
        W_t = W_knn @ W_t
        W_geo[W_t==0] += 1
        i += 1
    return W_geo



def geo_diffs(X, Y = [], k = 25):
    N_x = max(list(X.shape))
    N_y = 0
    if len(Y):
        N_y = max(list(Y.shape))
        XY = np.concatenate((X,Y), axis = 1)
        Z = deepcopy(XY)
    else:
        Z = deepcopy(X)
    eta_args = {'k': k}
    W = unif_diffs(torch.tensor(Z))[0].detach().cpu().numpy()
    W_knn = eta_knn(W, eta_args)
    W_t = normalize_rows(W_knn)
    W_diff = get_W_geo(W_t, W_knn)
    if N_y:
        W_diff_XX = W_diff[:N_x, :N_x]
        W_diff_XY = W_diff[:N_x, N_x:]
        W_dif_YY = W_diff[N_x:, N_x:]
        return W_diff_XX ,W_diff_XY,W_dif_YY
    return W_diff




def boosted_geo_diffs(X, Y = [], k = 25, m = 2):
    N_x = max(list(X.shape))
    unif_params = {'Y': X, 'print_freq': 100, 'learning_rate': 1,
                   'diff_map': unif_diffs, 'diff_quantiles': [0, .4]}
    X_res = dict_to_np(get_res_dict(X, unif_params))
    alpha = X_res['alpha']
    XU = resample(X, alpha, N=m * N_x)
    W_XUXU, W_XUX, W_XX = geo_diffs(XU, X, k = k)
    return W_XX




def run():
    N = 400
    X = normal_theta_circle(N)
    unif_params = {'Y': X, 'print_freq': 100, 'learning_rate': 1,
                   'diff_map': unif_diffs, 'diff_quantiles': [0,.4]}
    X_res = dict_to_np(get_res_dict(X, unif_params))
    alpha = X_res['alpha']
    XU = resample(X, alpha, N = 3 * N)

    W_XUXU ,W_XUX,W_XX = geo_diffs(XU, X)

    WXX_theta = circle_diffs(torch.tensor(X), torch.tensor(X))[0].detach().cpu().numpy()
    WXX_unif = unif_diffs(torch.tensor(X), torch.tensor(X))[0].detach().cpu().numpy()


    diff_ratios = np.log((W_XX/WXX_theta).flatten())
    sorted_ratios = np.sort(diff_ratios)
    plt.plot(sorted_ratios)
    plt.savefig('geo_diff_rationsXX.png')
    clear_plt()


    unif_diff_ratios = np.log(((WXX_unif/WXX_theta))).flatten()
    sorted_ratios = np.sort(unif_diff_ratios)
    plt.plot(sorted_ratios)
    plt.savefig('unif_diff_rationsXX.png')
    clear_plt()





if __name__=='__main__':
    run()
