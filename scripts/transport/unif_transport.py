import numpy as np
import torch.nn as nn
import torch
from transport_kernel import k_matrix, normalize, radial_kernel
import matplotlib.pyplot as plt
import time
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import least_squares
import os
from ellipse import rand_ellipse
import pandas as pd
from get_data import normal_theta_circle

def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True


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



def hull_sample(X,Y = [], n_samples = 1000):
    if not len(Y):
        Y = X
    N = len(X.T)
    X_indexes = np.random.randint(0, N, size = n_samples)
    M = len(Y.T)
    Y_indexes = np.random.randint(0, M, size = n_samples)
    factors = 2 * np.random.random(size = n_samples)

    return X[:,X_indexes]*factors + Y[:,Y_indexes]*(2-factors)


def hull_embedding(Y,X = [], nh_samples = 1000, hull_ratio = .2,
                   just_Y = False, just_hull = False):
    N = len(Y.T)
    if not len(X):
        X = Y[:]
    hull_samples = hull_sample(Y,X, n_samples = nh_samples)
    if just_Y:
        X_resampled = resample(X, N=1)
        Y_resampled = resample(Y, N=int(nh_samples // (2 * hull_ratio)))
    elif just_hull:
        X_resampled = resample(X, N=1)
        Y_resampled =   resample(Y, N=1)
    else:
        X_resampled = resample(X, N=int(nh_samples // (2 * hull_ratio)))
        Y_resampled = resample(Y, N=int(nh_samples // (2 * hull_ratio)))
    return np.concatenate([hull_samples, Y_resampled, X_resampled], axis = 1)


def l_scale(X):
    if X.shape[1] > 1:
        return float(torch.quantile(k_matrix(X,X)))
    return float(torch.quantile(k_matrix(X, X), q=.25))


def get_theta(X,Y):
    Y = Y.reshape(len(Y))
    X = X.reshape(len(X))
    thetas = torch.arctan2(Y , X)
    return thetas + 3.14159265359


def unif_diffs(sample, sample_alt =[]):
    if not len(sample_alt):
        sample_alt = sample
    diffs = k_matrix(torch.tensor(sample_alt.T), torch.tensor(sample.T))
    return diffs, sample


def circle_diffs(sample, sample_alt = []):
    if not len(sample_alt):
        sample_alt = sample
    X,Y = sample[0], sample[1]
    thetas =  get_theta(X,Y)
    thetas = thetas.reshape(len(thetas), 1)

    X_alt, Y_alt = sample_alt[0], sample_alt[1]
    thetas_alt = get_theta(X_alt, Y_alt)
    thetas_alt = thetas_alt.reshape(len(thetas_alt), 1)

    diffs = k_matrix(thetas_alt, thetas)
    diffs_2pi = 2*np.pi - diffs
    diffs = torch.min(diffs,  diffs_2pi)
    return diffs, thetas, thetas_alt


def sort_rank(sorted_vec,  val):
    ks = (sorted_vec == val).nonzero()
    return len(ks) + ks[0]


def indexing_tensor(shape):
    zero_tensor = torch.zeros(shape)
    idx_tensor = (zero_tensor == 0).nonzero()
    return idx_tensor


def get_rharmonics(N):
    harmonics = np.zeros(N)
    h_sum = 0
    for i in range(1,N+1):
        h_sum += 1/i
        harmonics[i-1] += h_sum
    return np.flip(harmonics)


def W_inf(W):
    n = len(W)
    c_factor = (1/(1+(1/n)))
    w_min = torch.min(W[W > 0])
    c_min = w_min * c_factor

    log_max = torch.log((torch.max(W) + c_min))
    return log_max - torch.log(W + c_min)


def W_inf_range(W, a,b):
    W = torch.tensor(W)
    a = torch.tensor(a)
    b = torch.tensor(b)
    n = len(W)
    c_factor = (1/(1+(1/n)))
    c_min = a * c_factor

    log_min = torch.log(c_min)
    log_max = torch.log((b + c_min))

    W_inf = log_max - torch.log(W + c_min)
    W_inf[W < a] = log_max - log_min
    W_inf[W > b] = 0
    return W_inf


class UnifKernel(nn.Module):
    def __init__(self, base_params, device=None):
        super().__init__()
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        self.dtype = torch.float32
        base_params['device'] = self.device
        self.params = base_params
        self.diff_map = self.params['diff_map']
        a, b = self.params['diff_quantiles']
        self.Y = torch.tensor(base_params['Y'], device = self.device, dtype = self.dtype)

        self.W = torch.tensor(self.diff_map(self.Y)[0])
        self.thetas = torch.tensor(self.diff_map(self.Y)[1])

        a,b = self.params['diff_quantiles']
        self.a_qval = torch.quantile(self.W[self.W > 0], q = a)
        self.b_qval = torch.quantile(self.W, q = b)
        self.W_rank = W_inf_range(self.W, self.a_qval, self.b_qval)
        self.N = len(self.W)

        if len(self.params['Y_tilde']):
            self.Y_tilde = torch.tensor(base_params['Y_tilde'], device=self.device, dtype=self.dtype)
            self.W_tilde = torch.tensor(self.diff_map( self.Y_tilde, self.Y)[0])

            self.a_qval = torch.max(torch.min(self.W_tilde, dim=1)[0])
            self.b_qval = torch.quantile(self.W_tilde[:1500, :10000], q=.3)
            self.W_tilde_rank = W_inf_range(self.W_tilde, self.a_qval, self.b_qval)

        self.iters = 0
        if 'target' in self.params.keys():
            self.target_vec = (self.N)*torch.tensor(self.params['target'], device = self.device, dtype = self.dtype)
        else:
            self.target_vec = (self.N)*torch.ones(self.N, device = self.device, dtype = self.dtype)


def f_exp(alpha, W_rank, target):
    y = np.dot(W_rank, np.exp(alpha)) - target
    return np.dot(y, y)


def grad_f_exp(alpha, W_rank, target):
    exp_alpha = np.exp(alpha)
    dx = np.diag(exp_alpha)
    dz = 2 * (W_rank_2 @ exp_alpha - W_rank_t)
    grad = dz @ dx
    return grad


def resample(Y, alpha = [], N = 10000):
    n = len(Y.T)
    if not len(alpha):
        alpha = np.full(n, 1/n)
    resample_indexes = np.random.choice(np.arange(n), size=N, replace=True, p=alpha)
    Y_resample = Y[:, resample_indexes]
    return Y_resample


def normalize_cols(W):
    return np.diag(np.diag(W @ W.T)**-1) @ W


def normalize_rows(W):
    for i,row in enumerate(W):
        W[i] *= 1/np.linalg.norm(W[i], ord = 1)
    return W


def inverse_smoothing(alpha, W, l = .08):
    alpha_inv = 1/alpha
    smoothing = normalize_rows(np.exp(-np.abs(W - np.diag(W)) / l))
    alpha_inv_smooth = smoothing @ alpha_inv
    return one_normalize(1/alpha_inv_smooth)


def smoothing(alpha, W, l = .08):
    smoothing = normalize_rows(np.exp(-np.abs(W)/ l))
    alpha_smooth = smoothing @ alpha
    return one_normalize(alpha_smooth)


def one_normalize(vec):
    return vec/np.linalg.norm(vec, ord = 1)


def one_normalize_trunc(vec, q = .4):
    thresh = np.quantile(vec, q)
    vec[vec < thresh] = 0
    return one_normalize(vec)


def get_inverse_res_dict(Y,Y_tilde, params):
    params['Y_tilde'] = Y_tilde
    Y_model = UnifKernel(params)

    W = np.asarray(Y_model.W.cpu())
    W_rank = np.asarray(Y_model.W_rank.cpu())

    W_tilde_rank = np.asarray(Y_model.W_tilde_rank.cpu())
    W_rank_2 = W_tilde_rank.T @ W_tilde_rank

    Y_target =  np.asarray(Y_model.target_vec.cpu())
    target = W_rank @ Y_target
    target = one_normalize(smoothing(target,  W, l = .05))
    W_rank_t = W_tilde_rank.T @ target

    def f(alpha):
        y = np.dot(W_tilde_rank, alpha) - target
        return np.dot(y, y)

    def grad_f(alpha):
        return 2 * (W_rank_2 @ alpha - W_rank_t)

    N = len(Y_tilde.T)
    x_0 = np.full(N, 1 / N)

    bnds = [(1 / N ** 2, np.inf) for i in range(N)]
    result = minimize(f, x_0, method='L-BFGS-B', bounds=bnds, jac=grad_f,
                      options={'disp': False, 'maxiter': 10000, 'maxfun': 500000, 'gtol': 1e-11, 'ftol': 1e-11})

    alpha = result['x']
    alpha = one_normalize(alpha).reshape(len(alpha))
    alpha_inv = one_normalize(1 / alpha).reshape(len(alpha))

    inverse_res_dict = {'alpha': alpha, 'alpha_inv': alpha_inv,
                'W': W, 'W_rank': W_rank, 'model' : Y_model}
    return inverse_res_dict


def get_res_dict(Y,params):
    params['Y_tilde'] = []
    Y_model = UnifKernel(params)

    W = np.asarray(Y_model.W.cpu())
    W_rank = np.asarray(Y_model.W_rank.cpu())
    W_rank_2 = W_rank @ W_rank
    target = np.asarray(Y_model.target_vec.cpu())
    W_rank_t = W_rank @ target

    def f(alpha):
        y = np.dot(W_rank, alpha) - target
        return np.dot(y, y)

    def grad_f(alpha):
        return 2 * (W_rank_2 @ alpha - W_rank_t)

    N = len(Y.T)
    x_0 = np.full(N, 1 / N)

    bnds = [(1 / N ** 2, np.inf) for i in range(N)]
    result = minimize(f, x_0, method='L-BFGS-B', jac=grad_f, bounds=bnds,
                      options={'disp':False, 'maxiter': 10000, 'maxfun': 500000, 'gtol': 1e-11, 'ftol': 1e-11})

    alpha = result['x']
    alpha = one_normalize(alpha).reshape(len(alpha))
    alpha_smooth = smoothing(alpha, W, l=.1)

    alpha_inv = one_normalize(1 / alpha).reshape(len(alpha))
    alpha_inv_smooth = inverse_smoothing(alpha_inv, W, l=.1)

    res_dict = {'alpha': alpha, 'alpha_smooth': alpha_smooth,
                'alpha_inv': alpha_inv, 'alpha_inv_smooth': alpha_inv_smooth,
                'W': W, 'W_rank': W_rank, 'model' : Y_model}
    return res_dict


def run():
    N = 400
    diff_quantiles = [0,.4]
    Y = normal_theta_circle(1000)
    diff_map = circle_diffs
    params = {'Y': Y, 'print_freq': 1000, 'learning_rate': 1, 'lambda_reg': 1e-1, 'Y_tilde': [],
                   'nugget': 1e-3, 'diff_map': diff_map, 'diff_quantiles': diff_quantiles}
    res_dict = get_res_dict(Y, params)
    alpha = res_dict['alpha']
    Y_resample = resample(Y, alpha , N)
    sample_hmap(Y_resample, 'Y_hmmm.png')


if __name__=='__main__':
    run()



