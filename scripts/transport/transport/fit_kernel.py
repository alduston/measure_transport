import torch
import numpy as np
from transport_kernel import  TransportKernel, l_scale, normalize,get_kernel
from regress_kernel import RegressionKernel, UnifKernel2
import matplotlib.pyplot as plt
import os
from unif_transport import get_res_dict, smoothing, unif_diffs, one_normalize, circle_diffs, geo_circle_diffs
from get_data import resample, normal_theta_circle, normal_theta_two_circle, sample_normal,\
    sample_swiss_roll, sample_moons, sample_rings, sample_circles,sample_banana, sample_spirals, \
    normal_theta_circle_noisy,sample_pinweel,sample_unif_dumbell

from picture_to_dist import sample_elden_ring, sample_bambdad, sample_twisted_rings
from kernel_geodesics import geo_diffs, boosted_geo_diffs
from copy import deepcopy
import random

import warnings
warnings.filterwarnings("ignore")


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True


def update_list_dict(Dict, update):
    for key, val in update.items():
        Dict[key].append(val)
    return Dict


def prob_normalization(alpha):
    N = len(alpha)
    c_norm = torch.log(N / torch.sum(torch.exp(alpha)))
    return alpha - c_norm


def train_kernel(kernel_model, n_iter = 100):
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr=kernel_model.params['learning_rate'])
    Loss_dict = {'n_iter': [], 'fit': [], 'reg': [], 'total': []}
    kernel_model.train()
    for i in range(n_iter):
        loss, loss_dict = train_step(kernel_model, optimizer)
        if not i % kernel_model.params['print_freq']:
            print(f'At step {i}: fit_loss = {round(float((loss_dict["fit"])),6)},'
                  f' reg_loss = {round(float(loss_dict["reg"]),6)}')
            Loss_dict = update_list_dict(Loss_dict, loss_dict)
    return kernel_model, Loss_dict


def train_step(kernel_model, optimizer):
    optimizer.zero_grad()
    loss, loss_dict = kernel_model.loss()
    loss.backward()
    optimizer.step()

    kernel_model.iters += 1
    loss_dict['n_iter'] = kernel_model.iters
    return loss, loss_dict


def sample_hmap(sample, save_loc, bins = 20, d = 2, range = None, vmax= None, cmap = None):
    try:
        sample = sample.detach().cpu()
    except AttributeError:
        pass
    if d == 2:
        x, y = sample.T
        x = np.asarray(x)
        y = np.asarray(y)
        plt.hist2d(x,y, density=True, bins = bins, range = range, cmin = 0, vmin=0, vmax = vmax, cmap = cmap)
        plt.colorbar()
    elif d == 1:
        x =  sample
        x = np.asarray(x)
        plt.hist(x, bins = bins, range = range)
    plt.savefig(save_loc)
    clear_plt()
    return True


def sample_scatter(sample, save_loc, bins = 20, d = 2, range = None):
    try:
        sample = sample.detach().cpu()
    except AttributeError:
        pass
    x, y = sample.T
    x = np.asarray(x)
    y = np.asarray(y)
    plt.scatter(x,y, s=7)
    
    if range != None:
        x_left, x_right = range[0]
        y_bottom, y_top = range[1]
        plt.xlim(x_left, x_right)
        plt.ylim(y_bottom, y_top)
    plt.savefig(save_loc)
    clear_plt()
    return True

def dict_to_np(dict):
    for key,val in dict.items():
        try:
            dict[key] = val.detach().cpu().numpy()
        except BaseException:
            pass
    return dict


def unif_boost_exp(Y_gen, X_gen = None, exp_name= 'exp',
                   N = 500, n_bins = 30,plt_range = None, t_iter = 501,
                   vmax = None, q = 0, s = 1.00, r_diff_map = unif_diffs):
    save_dir = f'../../data/kernel_transport/{exp_name}'

    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    d = 2

    Y = torch.tensor(Y_gen(N),  device = device)
    Y_test = torch.tensor(Y_gen(N), device=device)
    if Y.shape[0] > Y.shape[1]:
        Y = Y.T
        Y_test = Y_test.T
    N = len(Y.T)

    tilde_scale = 5 * N

    sample_hmap(Y.T, f'{save_dir}/Y_hmap.png', d=d, bins= n_bins, range=plt_range, vmax = vmax)
    sample_scatter(Y.T, f'{save_dir}/Y_scatter.png', d=d, bins= n_bins, range=plt_range)

    l_y = l_scale(Y.T)
    fit_params = {'name': 'radial', 'l': l_y/7, 'sigma': 1}
    mmd_params = {'name': 'radial', 'l': l_y/9, 'sigma': 1}


    unif_params = {'Y': Y.T, 'fit_kernel_params': fit_params,
     'mmd_kernel_params': mmd_params, 'normalize': False, 'one_lambda': 5,
     'reg_lambda': 7e-6, 'print_freq': 100, 'learning_rate': .01,
     'nugget': 1e-4}


    unif_kernel = UnifKernel2(unif_params)
    train_kernel(unif_kernel, n_iter= 5 * t_iter)
    alpha_y = one_normalize(unif_kernel.get_alpha_p().detach().cpu().numpy() ** s)
    alpha_y = alpha_y.reshape(len(alpha_y))


    Y_unif = resample(Y, alpha_y, N).reshape(Y.shape)
    Y_unif1 = resample(Y, alpha_y, tilde_scale)
    Y_unif2 = resample(Y, alpha_y, 10 * tilde_scale)

    unif_thetas = circle_diffs(Y_unif1)[1].detach().cpu().numpy().reshape(tilde_scale)
    plt.hist(unif_thetas)
    plt.savefig('unif_thetas.png')
    clear_plt()


    if X_gen == None:
        X = (Y_unif + torch.tensor(sample_normal(N, d), device=device).reshape(Y.shape)).T
        X1 = (Y_unif1 + torch.tensor(sample_normal(tilde_scale, d), device=device).reshape(Y_unif1.shape)).T

    else:
        X = torch.tensor(X_gen(N), device=device)
        X1 = torch.tensor(X_gen(tilde_scale), device=device)

    X2 = sample_normal(N, d)
    X3 = sample_normal(tilde_scale, d)

    l = l_scale(X)
    fit_params['l'] = l/7
    mmd_params['l'] = l/7


    sample_hmap(X, f'{save_dir}/X_hmap.png', d=d, bins= n_bins)
    sample_hmap(Y_unif2.T, f'{save_dir}/Y_unif_hmap.png', d=d, bins= n_bins, range=plt_range, vmax = None)
    sample_scatter(Y_unif2.T, f'{save_dir}/Y_unif_scatter.png', d=d, bins=n_bins, range=plt_range)


    basic_transport_params = {'X': X2, 'Y': Y.T, 'fit_kernel_params': fit_params,
                        'mmd_kernel_params': mmd_params, 'normalize': False,'one_lambda': 5,
                        'reg_lambda': 1e-5, 'unif_lambda': 0, 'print_freq': 100, 'learning_rate': .1,
                        'nugget': 1e-4, 'X_tilde': X1, 'alpha_y': [], 'alpha_x': False}


    basic_transport_kernel = TransportKernel(basic_transport_params)
    train_kernel(basic_transport_kernel, n_iter= t_iter)

    Y_pred = basic_transport_kernel.map(X2).detach().cpu().numpy()
    Y_pred1 = basic_transport_kernel.map(X3).detach().cpu().numpy()

    unif_transport_params = deepcopy(basic_transport_params)
    unif_transport_params['alpha_y'] = alpha_y
    unif_transport_kernel = TransportKernel(unif_transport_params)
    train_kernel(unif_transport_kernel, n_iter=t_iter)

    dual_transport_params = deepcopy(basic_transport_params)
    dual_transport_params['alpha_x'] = True
    dual_transport_params['X'] = X2
    dual_transport_params['reg_lambda_alpha'] = 1e-7
    dual_transport_params['learning_rate'] = .01

    dual_transport_kernel = TransportKernel(dual_transport_params)
    train_kernel(dual_transport_kernel, n_iter= 5 * t_iter)
    Y_pred_dual = dual_transport_kernel.map(X2).detach().cpu().numpy()


    #Y_ulatent_pred_dual = dual_transport_kernel.Z.reshape(Y_pred_dual.shape) + X.reshape(Y_pred_dual.shape)
    #sample_hmap(Y_ulatent_pred_dual.T, f'{save_dir}/Y_ulatent_pred_dual.png', d=d, bins=n_bins, range=plt_range, vmax=None)

    Y_ulatent_pred = unif_transport_kernel.map(X).detach().cpu().numpy()
    Y_ulatent_pred1 = unif_transport_kernel.map(X1).detach().cpu().numpy()
    sample_hmap(Y_ulatent_pred1.T, f'{save_dir}/Y_ulatent_pred.png', d=d, bins= n_bins, range=plt_range, vmax=None)
    sample_scatter(Y_ulatent_pred1.T, f'{save_dir}/Y_ulatent_pred_scatter.png', d=d, bins=n_bins, range=plt_range)


    lr = l_scale(torch.tensor(Y_ulatent_pred.T))
    r_fit_params = {'name': 'radial', 'l': lr/7, 'sigma': 1}
    r_mmd_params = {'name': 'radial', 'l': lr/7, 'sigma': 1}

    regression_params = {'Y': Y.T, 'Y_unif': Y_ulatent_pred1.T, 'fit_kernel_params': r_fit_params, 'one_lambda': 5,
                         'reg_lambda': 1e-7, 'mmd_kernel_params': r_mmd_params, 'print_freq': 500, 'diff_map': r_diff_map,
                          'learning_rate': .03, 'nugget': 1e-4}


    naive_regression_params = deepcopy(regression_params)
    #naive_regression_params['Y_unif'] = Y_pred1.T
    naive_regression_params['Y_unif'] = X1


    regression_kernel =  RegressionKernel(regression_params)
    train_kernel(regression_kernel, n_iter= 5 * t_iter)
    #alpha_inv = regression_kernel.map(Y_ulatent_pred1.T)
    alpha_inv = regression_kernel.map(Y_ulatent_pred1.T, Z_y  = regression_kernel.Z)
    Y_pred_unif = resample(Y_ulatent_pred1, alpha_inv, N= tilde_scale)


    naive_regression_kernel = RegressionKernel(naive_regression_params)
    train_kernel(naive_regression_kernel, n_iter= 5 * t_iter)
    alpha_inv_naive = naive_regression_kernel.map(X1.T, Z_y=naive_regression_kernel.Z)
    Y_pred_naive = resample(X1.T, alpha_inv_naive, N=tilde_scale)

    # alpha_inv_naive = naive_regression_kernel.map(Y_pred.T)
    # alpha_inv_naive = naive_regression_kernel.map(Y_pred1.T, Z_y=naive_regression_kernel.Z)
    # Y_pred_naive = resample(Y_pred1, alpha_inv_naive, N=tilde_scale)


    if q:
        Y = (Y.T[Y[0] < q][:N]).T
        Y_pred = (Y_pred.T[Y_pred[0] < q][:N]).T
        Y_pred_dual = (Y_pred_dual.T[Y_pred_dual[0] < q][:N]).T
        Y_pred_unif = (Y_pred_unif.T[Y_pred_unif[0] < q][:N]).T
        Y_pred_naive = (Y_pred_naive.T[Y_pred_naive[0] < q][:N]).T
        Y_test = (Y_test.T[Y_test[0] < q][:N]).T


    sample_hmap(Y_pred.T, f'{save_dir}/Y_pred_hmap_{N}.png', d=d, bins= n_bins, range=plt_range, vmax=vmax)
    sample_hmap(Y_pred_unif.T, f'{save_dir}/Y_pred_unif_hmap_{N}.png', d=d, bins= n_bins, range=plt_range, vmax=vmax)
    sample_hmap(Y_pred_naive.T, f'{save_dir}/Y_pred_naive_hmap_{N}.png', d=d, bins=n_bins, range=plt_range, vmax=vmax)
    sample_hmap(Y_pred_dual.T, f'{save_dir}/Y_pred_dual_hmap_{N}.png', d=d, bins= n_bins, range=plt_range, vmax=vmax)

    sample_scatter(Y_pred.T, f'{save_dir}/Y_pred_scatter_{N}.png', d=d, bins=n_bins, range=plt_range)
    sample_scatter(Y_pred_unif.T, f'{save_dir}/Y_pred_unif_scatter_{N}.png', d=d, bins= n_bins, range=plt_range)
    sample_scatter(Y_pred_naive.T, f'{save_dir}/Y_pred_naive_scatter_{N}.png', d=d, bins=n_bins, range=plt_range)
    sample_scatter(Y_pred_dual.T, f'{save_dir}/Y_pred_dual_scatter_{N}.png', d=d, bins= n_bins, range=plt_range)

    Y = torch.tensor(Y, device=device)
    Y_pred = torch.tensor(Y_pred, device=device)
    Y_pred_dual = torch.tensor(Y_pred_dual, device=device)
    Y_pred_naive = torch.tensor(Y_pred_naive, device=device)
    Y_pred_unif = torch.tensor(Y_pred_unif, device=device)

    #res_thetas = circle_diffs(Y_pred_unif)[1].detach().cpu().numpy().reshape(len(Y_pred_unif.T))
    #plt.hist(res_thetas)
    #plt.savefig('res_thetas.png')
    #clear_plt()

    mmd_vanilla =  basic_transport_kernel.mmd(map_vec = Y_pred.T, target = Y_test.T)
    mmd_dual = basic_transport_kernel.mmd(map_vec = Y_pred_dual.T, target = Y_test.T)
    mmd_naive = basic_transport_kernel.mmd(map_vec=Y_pred_naive.T, target=Y_test.T)
    mmd_unif = basic_transport_kernel.mmd(map_vec = Y_pred_unif.T, target = Y_test.T)
    mmd_opt = basic_transport_kernel.mmd(map_vec = Y.T, target = Y_test.T)

    return mmd_vanilla,mmd_dual, mmd_unif, mmd_opt,mmd_naive


def banana_exp(N = 1500, q = 0):
    plt_range = [[-4, 4], [-1, 10]]
    vmax = .5
    Y_gen = sample_banana
    X_gen = None
    exp_name = 'banana_test'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=501, n_bins=30, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def ring_exp(N = 1500, q = 0):
    plt_range = [[-4.2, 4.2], [-4.2, 4.2]]
    vmax = .5
    Y_gen = sample_rings
    X_gen = None
    exp_name = 'ring_test'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=501, n_bins=30, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def twisted_ring_exp(N = 1500, q = 0):
    plt_range = [[-1.1, 1.1], [-1.1, 1.1]]
    vmax = None
    Y_gen = sample_twisted_rings
    X_gen = None
    exp_name = 'twisted_ring'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=501, n_bins=30, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')



def moons_exp(N = 1500, q = None):
    plt_range = [[-3.5, 3.5], [-3.5, 3.5]]
    vmax = .33
    Y_gen = sample_moons
    X_gen = None
    exp_name = 'moons_test'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=601, n_bins=40, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def pinweel_exp(N = 1500, q = 0):
    plt_range = [[-3.3, 3.3], [-3.3, 3.3]]
    vmax = .5
    Y_gen = sample_pinweel
    X_gen = None
    exp_name = 'pinweel_test'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=801, n_bins=30, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def swiss_roll_exp(N = 1500, q = 0):
    plt_range = [[-3.5, 3.5], [-3.5, 3.5]]
    vmax = .35
    Y_gen = sample_swiss_roll
    X_gen = None
    exp_name = 'swiss_roll_test'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=501, n_bins=30, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def spiral_exp(N= 1500, q = 0):
    plt_range = [[-3.5, 3.5], [-3.5, 3.5]]
    vmax = .3
    Y_gen = sample_spirals
    X_gen = None
    exp_name = 'spiral_test'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=501, n_bins=30, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def elden_exp(N = 10000, q = 0):
    plt_range = [[-1, 1], [-1.2, 1.2]]
    vmax = 5
    Y_gen = sample_elden_ring
    X_gen = None
    exp_name = 'elden4'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=1501, n_bins=70, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def two_circle_exp(N = 1000, q = 0):
    plt_range = [[-1.5, 1.5], [-3.5, 3.5]]
    vmax = None
    Y_gen = normal_theta_two_circle
    X_gen = None
    exp_name = 'two_circle_test'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=501, n_bins=30, s=1, q=q)

    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def bambdad_exp(N = 8000, q = 0):
    plt_range = [[-1.5, 1.5], [-1.5, 1.5]]
    vmax = None
    Y_gen = sample_bambdad
    X_gen = None
    exp_name = 'bambdad4'
    mmd_vanilla, mmd_dual, mmd_unif, mmd_opt, mmd_naive = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range,n_bins=60,
                                                                         vmax=vmax, t_iter = 1001, q = q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def circle_exp(N = 1000, q = 0):
    plt_range = [[-1.5, 1.5], [-1.5, 1.5]]
    vmax = None
    Y_gen = normal_theta_circle
    X_gen = sample_normal
    exp_name = 'circle_test'
    mmd_vanilla,mmd_dual, mmd_unif, mmd_opt,mmd_naive =  unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=801, n_bins=30, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def gen_exp(N, Y_gen, name,  plt_range = None, X_gen = sample_normal, vmax = None, q = None):
    exp_name = name
    mmd_vanilla,mmd_dual, mmd_unif, mmd_opt,mmd_naive =  unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=801, n_bins=30, s=1, q=q)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Dual mmd was {mmd_dual}')
    print(f'Uniform  mmd was {mmd_unif}')
    print(f'Naive mmd was {mmd_naive}')
    print(f'Optimal mmd was {mmd_opt}')

    save_dir = f'../../data/kernel_transport/{exp_name}'
    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}, opt: {mmd_opt}" > {save_dir}/mmd_results.txt ')
    os.system(f'echo "dual: {mmd_dual} ,naive: {mmd_naive}" >> {save_dir}/mmd_results.txt ')


def comparison_exp(Y_gen, name = '', q = None, plt_range =  None, X_gen = None, trials = 20):
    vmax = 8
    Ns =  [200, 400, 600, 800, 1000, 1200, 1600, 2000]
    exp_name = f'mmd_regression_test_{name}'
    save_dir = f'../../data/kernel_transport/{exp_name}'

    mean_unif_mmds = []
    mean_dual_mmds = []
    mean_vanilla_mmds = []
    mean_naive_mmds = []
    mean_opt_mmds = []

    for N in Ns:
        vanilla_mmds = []
        unif_mmds = []
        dual_mmds = []
        naive_mmds = []
        opt_mmds = []

        for i in range(trials):
            mmd_vanilla,mmd_dual, mmd_unif, mmd_opt,mmd_naive =  unif_boost_exp(Y_gen, X_gen, exp_name=exp_name,
                                                                         N=N, plt_range=plt_range, vmax=vmax,
                                                                         t_iter=501, n_bins=30, s=1, q=q)
            unif_mmds.append(float(mmd_unif.detach().cpu()))
            vanilla_mmds.append(float(mmd_vanilla.detach().cpu()))
            opt_mmds.append(float(mmd_opt.detach().cpu()))
            naive_mmds.append(float(mmd_naive.detach().cpu()))
            dual_mmds.append(float(mmd_dual.detach().cpu()))

            print(f'N = {N}, trial {i+1}, mmd_vanilla = {round(float((mmd_vanilla)),6)},'
                  f' mmd_unif = {round(float((mmd_unif)),6)}, mmd_opt =  {round(float((mmd_opt)),6)}'
                  f' mmd_dual = {round(float((mmd_dual)),6)}, mmd_naive = {round(float((mmd_naive)),6)}')

        mean_dual_mmds.append(np.mean(dual_mmds))
        mean_unif_mmds.append(np.mean(unif_mmds))
        mean_vanilla_mmds.append(np.mean(vanilla_mmds))
        mean_naive_mmds.append(np.mean(naive_mmds))
        mean_opt_mmds.append(np.mean(opt_mmds))

    plt.plot(Ns, np.log(mean_unif_mmds), label = 'Unif transport')
    plt.plot(Ns, np.log(mean_vanilla_mmds),  label = 'Vanilla transport')
    plt.plot(Ns, np.log(mean_dual_mmds), label='Dual transport')
    plt.plot(Ns, np.log(mean_naive_mmds), label='Naive transport')
    plt.plot(Ns, np.log(mean_opt_mmds), label='Optimal mmd')
    plt.xlabel('Sample size')
    plt.ylabel('Log MMD')
    plt.title('Test MMD for Unif v Vanilla Transport Maps')
    plt.legend()
    plt.savefig(f'{save_dir}/MMD_comparison.png')
    clear_plt()


def run():
   range = [[-3.5, 2.5], [-1.1,1.1]]
   gen_exp(2000, sample_unif_dumbell, 'dumbell', plt_range=range)



if __name__=='__main__':
    run()
