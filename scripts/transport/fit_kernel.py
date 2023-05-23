import torch
import numpy as np
from transport_kernel import  TransportKernel, l_scale, normalize
import matplotlib.pyplot as plt
import os
from unif_transport import get_res_dict, smoothing, unif_diffs, one_normalize,\
    one_normalize_trunc, circle_diffs, inverse_smoothing, alt_smoothing, W_inf_range
from get_data import resample, normal_theta_circle, normal_theta_two_circle, sample_normal,\
    sample_swiss_roll, sample_moons, sample_rings, sample_circles,sample_banana


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


def train_kernel_transport(kernel_model, n_iter = 100, save_dir = '', d = 1,plt_range = [[-3,3]]):
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr=kernel_model.params['learning_rate'])
    Loss_dict = {'n_iter': [], 'fit': [], 'reg': [], 'total': []}
    kernel_model.train()
    for i in range(n_iter):
        if not i % 10:
            Y_pred = kernel_model.map(kernel_model.X).detach().cpu().numpy().T
            sample_hmap(Y_pred, f'{save_dir}/Y_in_progress{0 if i==0 else ""}.png', d=d, bins=30, range= plt_range)
        loss, loss_dict = train_step(kernel_model, optimizer)
        if not i % kernel_model.params['print_freq']:
            print(f'At step {i}: fit_loss = {round(float(loss_dict["fit"]),4)},'
                  f' reg_loss = {round(float(loss_dict["reg"]),4)}')
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


def sample_hmap(sample, save_loc, bins = 20, d = 2, range = None, vmax= None):
    try:
        sample = sample.detach().cpu()
    except AttributeError:
        pass
    if d == 2:
        x, y = sample.T
        x = np.asarray(x)
        y = np.asarray(y)
        plt.hist2d(x,y, density=True, bins = bins, range = range, cmin = 0, vmin=0, vmax = vmax)
        plt.colorbar()
    elif d == 1:
        x =  sample
        x = np.asarray(x)
        plt.hist(x, bins = bins, range = range)
    plt.savefig(save_loc)
    clear_plt()
    return True


def sample_scatter(sample, save_loc, bins = 20, d = 2, range = []):
    try:
        sample = sample.detach().cpu()
    except AttributeError:
        pass
    x, y = sample.T
    x = np.asarray(x)
    y = np.asarray(y)
    plt.scatter(x,y)
    
    if range != None:
        x_left, x_right = range[0]
        y_bottom, y_top = range[1]
        plt.xlim(x_left, x_right)
        plt.ylim(y_bottom, y_top)
    plt.savefig(save_loc)
    clear_plt()
    return True


def transport_exp(X_gen, Y_gen, exp_name, N=1000, plt_range=None, t_iter = 400):
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass
    d = 2

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    X = normalize(torch.tensor(X_gen(N), device = device))
    Y = torch.tensor(Y_gen(N),  device = device).reshape(X.shape)

    sample_hmap(Y, f'{save_dir}/Y_hmap.png', d=d, bins=30, range=plt_range)
    sample_scatter(Y, f'{save_dir}/Y_scatter.png', d=d, bins=30, range=plt_range)
    sample_hmap(X, f'{save_dir}/X_hmap.png', d=d, bins=30, range=plt_range)

    X_tilde = normalize(torch.tensor(X_gen(N),  device = device))
    l = l_scale(X) / 5

    fit_kernel_params = {'name': 'radial', 'l': l, 'sigma': 1}
    mmd_kernel_params = {'name': 'radial', 'l': l, 'sigma': 1}

    model_params = {'X': X, 'Y': Y, 'fit_kernel_params': fit_kernel_params,
                    'mmd_kernel_params': mmd_kernel_params,'reg_lambda': 1e-5,
                    'unif_lambda': 0, 'print_freq': 1, 'learning_rate': .1,
                    'nugget': 1e-3, 'X_tilde': X_tilde, 'normalize': False}

    kernel_model = TransportKernel(model_params)
    train_kernel_transport(kernel_model, n_iter=t_iter, save_dir=save_dir, d=d, plt_range=plt_range)

    Y_tilde = kernel_model.map(X_tilde).detach().cpu().numpy()
    sample_hmap(Y_tilde.T, f'{save_dir}/Ypred_hmap.png', d=2, bins=30, range=plt_range)
    sample_scatter(Y_tilde.T, f'{save_dir}/Ypred_scatter.png', d=2, bins=30, range=plt_range)
    return True

def dict_to_np(dict):
    for key,val in dict.items():
        try:
            dict[key] = val.detach().cpu().numpy()
        except BaseException:
            pass
    return dict

def unif_boost_exp(Y_gen, X_gen = None, exp_name= 'exp', diff_map = unif_diffs,  N = 500,
                   plt_range = None, t_iter = 300, diff_quantiles = [0.0, 0.4], q = 1, vmax = None):
    save_dir = f'../../data/kernel_transport/{exp_name}'

    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    smoothing_l = .1
    d = 2
    tilde_scale = 10000

    Y = torch.tensor(Y_gen(N),  device = device)
    if Y.shape[0] > Y.shape[1]:
        Y = Y.T

    sample_hmap(Y.T, f'{save_dir}/Y_hmap.png', d=d, bins=30, range=plt_range, vmax = vmax)
    sample_scatter(Y.T, f'{save_dir}/Y_scatter.png', d=d, bins=30, range=plt_range)

    unif_params = {'Y': Y, 'print_freq': 1000, 'learning_rate': 1,
                   'diff_map': diff_map, 'diff_quantiles': diff_quantiles}
    Y_res =  dict_to_np(get_res_dict(Y, unif_params))

    alpha = one_normalize(Y_res['alpha'] ** 1)
    alpha_inv = one_normalize(Y_res['alpha_inv'] ** 1)
    Y_resample = resample(Y, alpha, N).reshape(Y.shape)
    W_resample, thetas, resample_thetas = diff_map(torch.tensor(Y, device=device), torch.tensor(Y_resample, device=device))

    alpha_resample_inv = one_normalize(smoothing(alpha_inv, W_resample, l=smoothing_l) + (1/N**2))
    Y_resample_inv = resample(Y_resample, alpha_resample_inv, N).reshape(Y.shape)

    #thetas = thetas.cpu().numpy().reshape(len(thetas))
    #resample_thetas = resample_thetas.cpu().numpy().reshape(len(resample_thetas))

    #sort_idx = np.argsort(thetas)
    #resample_sort_idx = np.argsort(resample_thetas)

    #thetas_sorted = thetas[sort_idx]
    #resample_thetas_sorted = resample_thetas[resample_sort_idx].reshape(len(resample_thetas))

    #plt.plot(thetas_sorted, alpha_inv[sort_idx].reshape(len(thetas)))
    #plt.savefig('theta_v_alpha_inv.png')
    #clear_plt()

    #plt.plot(resample_thetas_sorted, alpha_resample_inv[resample_sort_idx].reshape(len(resample_thetas)))
    #plt.savefig('theta_v_alpha_inv_resample.png')
    #clear_plt()

    #plt.plot(resample_thetas_sorted, alpha_resample_inv_alt[resample_sort_idx].reshape(len(resample_thetas)))
    #plt.savefig('theta_v_alpha_inv_resample_alt.png')
    #clear_plt()


    if X_gen == None:
        X = (Y_resample + torch.tensor(sample_normal(N,d), device = device).reshape(Y.shape)).T
        Y_tilde = resample(Y, alpha, tilde_scale)
        X_tilde = (Y_tilde + torch.tensor(sample_normal(tilde_scale ,d), device = device).reshape(Y_tilde.shape)).T
    else:
        X = torch.tensor(X_gen(N),  device = device)
        X_tilde =  torch.tensor(X_gen(tilde_scale),  device = device)


    l = l_scale(X)
    sample_hmap(X, f'{save_dir}/X_hmap.png', d=d, bins=30)
    sample_hmap(Y_resample.T, f'{save_dir}/Y_resample_hmap.png', d=d, bins=30, range=plt_range, vmax = vmax)
    sample_hmap(Y_resample_inv.T, f'{save_dir}/Y_resample_inv_hmap.png', d=d, bins=30, range=plt_range, vmax = vmax)
    sample_scatter(Y_resample.T, f'{save_dir}/Y_resample_scatter.png', d=d, bins=30, range=plt_range)

    fit_kernel_params = {'name': 'radial', 'l': l/7, 'sigma': 1}
    mmd_kernel_params = {'name': 'radial', 'l': l/7, 'sigma': 1}

    model_params = {'X': X, 'Y': Y_resample.T, 'fit_kernel_params': fit_kernel_params,
                    'mmd_kernel_params': mmd_kernel_params, 'normalize': False,
                    'reg_lambda': 1e-5, 'unif_lambda': 0, 'print_freq': 1, 'learning_rate': .1, 'nugget': 1e-3,
                    'X_tilde': X_tilde}

    naive_model_params = {'X': X, 'Y': Y.T, 'fit_kernel_params': fit_kernel_params,
                          'mmd_kernel_params': mmd_kernel_params, 'normalize':  False,
                          'reg_lambda': 1e-5, 'unif_lambda': 0, 'print_freq': 1, 'learning_rate': .1, 'nugget': 1e-3,
                          'X_tilde': X}

    kernel_model = TransportKernel(model_params)
    train_kernel_transport(kernel_model, n_iter=t_iter, save_dir=save_dir, d=d, plt_range=plt_range)

    naive_kernel_model = TransportKernel(naive_model_params)
    train_kernel_transport(naive_kernel_model, n_iter=t_iter, save_dir=save_dir, d=d, plt_range=plt_range)

    Y_tilde = kernel_model.map(X_tilde).detach().cpu().numpy()
    sample_hmap(Y_tilde.T, f'{save_dir}/Ypred_hmap.png', d=d, bins=30, range=plt_range, vmax = vmax)

    Y_tilde_naive = naive_kernel_model.map(X_tilde).detach().cpu().numpy()
    sample_hmap(Y_tilde_naive.T, f'{save_dir}/Ypred_naive_hmap.png', d=d, bins=30, range=plt_range, vmax = vmax)
    sample_scatter(Y_tilde_naive.T, f'{save_dir}/Ypred_naive_scatter.png', d=d, bins=30, range=plt_range)

    W_tilde = diff_map(torch.tensor(Y, device = device), torch.tensor(Y_tilde, device = device))[0].cpu().numpy()
    alpha_tilde_inv = one_normalize((smoothing(alpha_inv, W_tilde, l=smoothing_l) + (1/tilde_scale**2))**1.15)
    Y_tilde_resample = resample(Y_tilde, alpha_tilde_inv, N = tilde_scale)

    Y_alt = torch.tensor(Y_gen(tilde_scale), device=device)
    Y_alt = (Y_alt.T[Y_alt[0] < q][:N]).T
    sample_hmap(Y_alt.T, f'{save_dir}/Y_alt_hmap.png', d=d, bins=30, range=plt_range, vmax = vmax)

    Y_pred_naive = torch.tensor(resample(Y_tilde_naive, N = tilde_scale), device = device)
    Y_pred_naive = (Y_pred_naive.T[Y_pred_naive[0] < q][:N]).T
    sample_hmap(Y_pred_naive.T, f'{save_dir}/Y_alt_naive_hmap.png', d=d, bins=30, range=plt_range, vmax = vmax)

    Y_pred = torch.tensor(resample(Y_tilde, alpha_tilde_inv, N= tilde_scale), device = device)
    Y_pred = (Y_pred.T[Y_pred[0] < q][:N]).T
    sample_hmap(Y_pred.T, f'{save_dir}/Y_alt_unif_hmap.png', d=d, bins=30, range=plt_range, vmax = vmax)

    mmd_naive = kernel_model.loss_fit(Y_pred_naive.T, Y_alt.T)
    mmd_unif = kernel_model.loss_fit(Y_pred.T, Y_alt.T)

    sample_hmap(Y_tilde_resample.T, f'{save_dir}/Ypred_resampled_hmap.png', d=d, bins=30, range=plt_range, vmax = vmax)
    sample_scatter(Y_tilde_resample.T, f'{save_dir}/Ypred_resampled_scatter.png', d=d, bins=30, range=plt_range)
    return mmd_naive, mmd_unif


def run():
    plt_range = [[-1.5,1.5],[-1.5,1.5]]
    vmax = 8
    Ns = [100, 200, 300, 400, 500, 700, 900, 1200, 1600, 2000]
    MMD_naives = []
    MMD_unifs = []
    Y_gen = normal_theta_circle
    X_gen = None
    diff_map = circle_diffs
    exp_name = 'mmd_sample_test'
    n_trials = 20
    q = 1.01
    for N in Ns:
        MMD_naive = 0
        MMD_unif = 0
        for n in range(n_trials):
            mmd_naive, mmd_unif = unif_boost_exp(Y_gen, X_gen, exp_name = exp_name,
                                                 diff_map = diff_map, N  = N, q = q,
                                                 plt_range = plt_range, vmax = vmax)
            MMD_naive += mmd_naive
            MMD_unif += mmd_unif
        MMD_naives.append(MMD_naive/n_trials)
        MMD_unifs.append(MMD_unif/n_trials)

    plt.plot(Ns, MMD_naives, label='naive')
    plt.plot(Ns, MMD_unifs, label='unif')
    plt.xlabel('Sample size')
    plt.ylabel('MMD')
    plt.ylim(bottom = 0)
    plt.legend()
    plt.savefig(f'../../data/kernel_transport/{exp_name}/mmd_v_sample_size.png')






if __name__=='__main__':
    run()
