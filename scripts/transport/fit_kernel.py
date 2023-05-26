import torch
import numpy as np
from transport_kernel import  TransportKernel, l_scale, normalize
from regress_kernel import RegressionKernel
import matplotlib.pyplot as plt
import os
from unif_transport import get_res_dict, smoothing, unif_diffs, one_normalize, circle_diffs
from get_data import resample, normal_theta_circle, normal_theta_two_circle, sample_normal,\
    sample_swiss_roll, sample_moons, sample_rings, sample_circles,sample_banana, normal_theta_circle_noisy
from picture_to_dist import sample_elden_ring
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


def train_kernel(kernel_model, n_iter = 100):
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr=kernel_model.params['learning_rate'])
    Loss_dict = {'n_iter': [], 'fit': [], 'reg': [], 'total': []}
    kernel_model.train()
    for i in range(n_iter):
        loss, loss_dict = train_step(kernel_model, optimizer)
        if i and not i % kernel_model.params['print_freq']:
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
    train_kernel(kernel_model, n_iter=t_iter)

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


def unif_boost_exp(Y_gen, X_gen = None, exp_name= 'exp', diff_map = unif_diffs,  N = 500, n_bins = 30,
                   plt_range = None, t_iter = 401, diff_quantiles = [0.0, 0.4], vmax = None):
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
    tilde_scale = 2 * N

    Y = torch.tensor(Y_gen(N),  device = device)
    if Y.shape[0] > Y.shape[1]:
        Y = Y.T

    sample_hmap(Y.T, f'{save_dir}/Y_hmap.png', d=d, bins= n_bins, range=plt_range, vmax = vmax)
    sample_scatter(Y.T, f'{save_dir}/Y_scatter.png', d=d, bins= n_bins, range=plt_range)

    unif_params = {'Y': Y, 'print_freq': 1000, 'learning_rate': 1,
                   'diff_map': diff_map, 'diff_quantiles': diff_quantiles}
    Y_res =  dict_to_np(get_res_dict(Y, unif_params))

    alpha = one_normalize(Y_res['alpha'] ** 1)
    Y_resample = resample(Y, alpha, N).reshape(Y.shape)

    Y1 = resample(Y, alpha, tilde_scale)
    if X_gen == None:
        X = (Y_resample + torch.tensor(sample_normal(N, d), device=device).reshape(Y.shape)).T
        X1 = (Y1 + torch.tensor(sample_normal(tilde_scale, d), device=device).reshape(Y1.shape)).T
        X2 = (Y1 + torch.tensor(sample_normal(tilde_scale, d), device=device).reshape(Y1.shape)).T
    else:
        X = torch.tensor(X_gen(N), device=device)
        X1 = torch.tensor(X_gen(tilde_scale), device=device)
        X2 = torch.tensor(X_gen(tilde_scale), device=device)

    l = l_scale(X)
    sample_hmap(X, f'{save_dir}/X_hmap.png', d=d, bins= n_bins)
    sample_hmap(Y_resample.T, f'{save_dir}/Y_resample_hmap.png', d=d, bins= n_bins, range=plt_range, vmax = vmax)
    fit_params = {'name': 'radial', 'l': l/7, 'sigma': 1}
    mmd_params = {'name': 'radial', 'l': l/7, 'sigma': 1}

    unif_transport_params = {'X': X, 'Y': Y_resample.T, 'fit_kernel_params': fit_params,
                    'mmd_kernel_params': mmd_params, 'normalize': False,
                    'reg_lambda': 1e-5, 'unif_lambda': 0, 'print_freq': 100, 'learning_rate': .1, 'nugget': 1e-3,
                    'X_tilde': X1}

    unif_transport_kernel = TransportKernel(unif_transport_params)
    train_kernel(unif_transport_kernel, n_iter=t_iter)

    transport_params = {'X': X, 'Y': Y.T,  'fit_kernel_params': fit_params,
                          'mmd_kernel_params': mmd_params, 'normalize':  False,
                          'reg_lambda': 1e-5, 'unif_lambda': 0, 'print_freq': 100, 'learning_rate': .1, 'nugget': 1e-3,
                          'X_tilde': X1}

    transport_kernel = TransportKernel(transport_params)
    train_kernel(transport_kernel, n_iter=t_iter)
    Y_pred = transport_kernel.map(X1).detach().cpu().numpy()

    Y_unif = unif_transport_kernel.map(X1).detach().cpu().numpy()
    Y_unif1 = unif_transport_kernel.map(X2).detach().cpu().numpy()
    sample_hmap(Y_unif.T, f'{save_dir}/Y_unif_hmap.png', d=d, bins= n_bins, range=plt_range, vmax=vmax)

    r_fit_params = {'name': 'radial', 'l': l / 7 , 'sigma': 1}
    r_mmd_params = {'name': 'radial', 'l': l / 7, 'sigma': 1}
    regression_params = {'Y': Y.T, 'Y_unif': Y_unif.T, 'fit_kernel_params': r_fit_params, 'one_lambda': 5,
                         'reg_lambda': 1e-5,'mmd_kernel_params': r_mmd_params, 'print_freq': 500,
                         'alpha': alpha, 'learning_rate': .01, 'nugget': 1e-3, 'W_inf': Y_res['W_rank']}

    regression_kernel =  RegressionKernel(regression_params)
    train_kernel(regression_kernel, n_iter= 15 * t_iter)

    alpha_inv1 = regression_kernel.map(Y_unif1.T)
    Y_pred_unif = resample(Y_unif1, alpha_inv1, N=tilde_scale)

    sample_hmap(Y_pred_unif.T, f'{save_dir}/Y_pred_unif_hmap_{N}.png', d=d, bins= n_bins, range=plt_range, vmax=vmax)
    sample_hmap(Y_pred.T, f'{save_dir}/Y_pred_hmap_{N}.png', d=d, bins= n_bins, range=plt_range, vmax=vmax)

    sample_scatter(Y_pred_unif.T, f'{save_dir}/Y_pred_unif_scatter.png', d=d, bins= n_bins, range=plt_range)
    sample_scatter(Y_pred.T, f'{save_dir}/Y_pred_scatter.png', d=d, bins= n_bins, range=plt_range)

    Y = torch.tensor(Y, device=device)
    Y_pred = torch.tensor(Y_pred, device=device)
    Y_pred_unif = torch.tensor(Y_pred_unif, device=device)

    mmd_vanilla = transport_kernel.mmd(map_vec = Y_pred.T, target = Y.T)
    mmd_unif = transport_kernel.mmd(map_vec =Y_pred_unif.T, target = Y.T)

    return mmd_vanilla, mmd_unif

def banana_exp():
    N = 800
    plt_range = [[-4, 4], [-1, 10]]
    vmax = .5
    Y_gen = sample_banana
    X_gen = None
    diff_map = unif_diffs
    exp_name = 'banana_test'
    mmd_vanilla, mmd_unif = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name, diff_map=diff_map,
                                           N=N, plt_range=plt_range, vmax=vmax, t_iter = 401, n_bins=40)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Uniform mmd was {mmd_unif}')


def swiss_roll_exp():
    N = 750
    plt_range = [[-3.5, 3.5], [-3.5, 3.5]]
    vmax = .3
    Y_gen = sample_swiss_roll
    X_gen = None
    diff_map = unif_diffs
    exp_name = 'swiss_roll_test'
    mmd_vanilla, mmd_unif = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name, diff_map=diff_map,
                                           N=N, plt_range=plt_range, vmax=vmax, t_iter = 400, n_bins=40)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Uniform mmd was {mmd_unif}')


def elden_exp():
    N = 10000
    plt_range = [[-1.5, 1.5], [-1.5, 1.5]]
    vmax = 5
    Y_gen = sample_elden_ring
    X_gen = None
    diff_map = unif_diffs
    exp_name = 'elden_test'
    mmd_vanilla, mmd_unif = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name, diff_map=diff_map,
                                           N=N, plt_range=plt_range, vmax=vmax, t_iter = 1000, n_bins=70)
    print(f'Vanilla ELDEN mmd was {mmd_vanilla}')
    print(f'Uniform ELDEN mmd was {mmd_unif}')

    save_dir = f'../../data/kernel_transport/{exp_name}'

    os.sys(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}" > {save_dir}/mmd_results.txt ')
    return True


def two_circle_exp():
    N = 1000
    plt_range = [[-1.5, 1.5], [-3.5, 3.5]]
    vmax = 8
    Y_gen = normal_theta_two_circle
    X_gen = None
    diff_map = unif_diffs
    exp_name = 'two_circle_test'
    mmd_vanilla, mmd_unif = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name, diff_map=diff_map,
                                           N=N, plt_range=plt_range, vmax=vmax, t_iter = 501, n_bins=30)
    print(f'Vanilla ELDEN mmd was {mmd_vanilla}')
    print(f'Uniform ELDEN mmd was {mmd_unif}')


def circle_exp():
    N = 1000
    plt_range = [[-1.5, 1.5], [-1.5, 1.5]]
    vmax = 8
    Y_gen = normal_theta_circle_noisy
    X_gen = None
    diff_map = circle_diffs
    exp_name = 'circle_test'
    mmd_vanilla, mmd_unif = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name, diff_map=diff_map,
                                           N=N, plt_range=plt_range, vmax=vmax, t_iter = 501, n_bins=30)
    print(f'Vanilla mmd was {mmd_vanilla}')
    print(f'Uniform  mmd was {mmd_unif}')
    save_dir = f'../../data/kernel_transport/{exp_name}'

    os.system(f'echo "vanilla: {mmd_vanilla} ,unif: {mmd_unif}" > {save_dir}/mmd_results.txt ')
    return True


def circle_comparison_exp():
    plt_range = [[-1.5, 1.5], [-1.5, 1.5]]
    vmax = 8
    Ns =  [200, 400, 600, 800, 1000, 1200, 1600, 2000]
    trials = 10

    Y_gen = normal_theta_circle
    X_gen = sample_normal
    diff_map = circle_diffs
    exp_name = 'mmd_regression_test'
    save_dir = f'../../data/kernel_transport/{exp_name}'

    mean_unif_mmds = []
    mean_mmds = []

    for N in Ns:
        unif_mmds = []
        mmds = []
        for i in range(trials):
            mmd_vanilla, mmd_unif = unif_boost_exp(Y_gen, X_gen, exp_name=exp_name, diff_map=diff_map,
                                                   N=N, plt_range=plt_range, vmax=vmax)
            unif_mmds.append(float(mmd_unif.detach().cpu()))
            mmds.append(float(mmd_vanilla.detach().cpu()))

            print(f'N = {N}, trial {i+1}, mmd_vanilla = {round(float((mmd_vanilla)),6)},'
                  f' mmd_unif = {round(float((mmd_unif)),6)}')

        mean_mmds.append(np.mean(mmds))
        mean_unif_mmds.append(np.mean(unif_mmds))

    plt.plot(Ns, np.log10(mean_unif_mmds), label = 'Unif transport')
    plt.plot(Ns, np.log10(mean_mmds),  label = 'Vanilla transport')
    plt.xlabel('Sample size')
    plt.ylabel('Log10 MMD')
    plt.title('Test MMD for Unif v Vanilla Transport Maps')
    plt.legend()
    plt.savefig(f'{save_dir}/MMD_comperison.png')
    clear_plt()


def run():
    circle_exp()



if __name__=='__main__':
    run()
