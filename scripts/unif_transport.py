import numpy as np
import torch.nn as nn
import torch
from fit_kernel import sample_hmap, train_step, update_list_dict, clear_plt
from transport_kernel import k_matrix, normalize, radial_kernel, l_scale
import matplotlib.pyplot as plt
import time
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from scipy.optimize import least_squares


def get_theta(X,Y):
    Y = Y.reshape(len(Y))
    X = X.reshape(len(X))
    thetas = np.arctan2(Y , X)
    return np.pi + thetas


def unif_circle(N = 200):
    theta = np.random.uniform(low = -np.pi, high = np.pi, size = N)
    X = np.cos(theta)
    Y = np.sin(theta)
    sample = np.asarray([[x,y] for x,y in zip(X,Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample


def proj_circle(N = 300):
    X = np.random.uniform(low=-1, high=1, size=N)
    Y = np.sqrt(np.ones(len(X)) - X**2) * np.random.choice([-1.0,1.0], size=N)
    sample = np.stack([X,Y]).reshape((2, len(X)))
    return sample


def normal_proj_circle( N = 300):
    X = np.random.normal(loc=0, scale=1, size=10*N)
    X = X[np.abs(X) <= 1][:N]
    Y = np.sqrt(np.ones(len(X)) - X**2) * np.random.choice([-1.0,1.0], size=N)
    sample = np.stack([X,Y]).reshape((2, len(X)))
    return sample


def geo_diffs(sample):
    X,Y = sample[0], sample[1]
    theta =  get_theta(X,Y)
    theta = theta.reshape(len(theta), 1)
    diffs = k_matrix(theta, theta)
    diffs_2pi = 2*np.pi - diffs
    diffs = torch.min(diffs,  diffs_2pi)
    return diffs


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



def rank_W_weighted(W):
    W_flat = W.flatten()
    log_min = np.log(min(W_flat[W_flat > 0]))
    log_max = np.log(max(W_flat))

    W_ranks = torch.zeros(W_flat.shape, dtype=torch.float32)
    values, indexes = torch.topk(W_flat, k = len(W_flat))

    for i,index in enumerate(indexes):
        val  = values[i]
        if val:
            W_ranks[index] += log_max - np.log(val)
        else:
            W_ranks[index] +=  log_max - log_min
    W_ranks = W_ranks.reshape(W.shape)
    return W_ranks


def rank_W(W):
    W_flat = W.flatten()
    N = len(W.flatten())
    W_ranks = torch.zeros(W_flat.shape, dtype=torch.float32)
    values, indexes = torch.topk(W_flat, k = len(W_flat))
    for i,index in enumerate(indexes):
        val  = values[i]
        if val:
            W_ranks[index] += i
        else:
            W_ranks[index] += N
    W_ranks = W_ranks.reshape(W.shape)
    return W_ranks


def rank_W_range(W, range = [0,.95]):

    W_flat = W.flatten()
    l = torch.quantile(W_flat,q=range[0])
    h = torch.quantile(W_flat,q=range[1])

    N = len(W.flatten())
    N_inner = int(.5 * N)

    W_ranks = torch.zeros(W_flat.shape, dtype=torch.float32)
    values, indexes = torch.topk(W_flat, k = len(W_flat))
    for i,index in enumerate(indexes):
        val  = values[i]
        if val <= l:
            W_ranks[index] += N_inner
        elif val >= h:
            W_ranks[index] += 0
        else:
            W_ranks[index] += i
    W_ranks = W_ranks.reshape(W.shape)
    return W_ranks


class UnitKernel(nn.Module):
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

        self.Y = torch.tensor(base_params['Y'], device = self.device, dtype = self.dtype)

        self.W = self.diff_map(self.Y)
        self.W_rank = rank_W_weighted(self.W)

        self.N = len(self.W)

        self.iters = 0
        self.alpha = nn.Parameter(self.init_alpha(), requires_grad=True)
        self.target_vec = (self.N)*torch.ones(self.N, device = self.device, dtype = self.dtype)


    def W_eps(self, eps):
        return (self.W>eps).float()


    def init_alpha(self):
        return torch.randn(self.N, device = self.device, dtype = self.dtype)


    def loss(self):
        alpha_l1 = torch.sum(torch.abs(self.alpha))
        alpha_proj = torch.abs(self.alpha/alpha_l1)
        loss = torch.mean((self.W_rank @ alpha_proj - self.target_vec)**2)
        loss_dict = {'square_loss': loss}
        return loss,loss_dict


    def get_alpha(self):
        alpha = torch.linalg.solve(self.W_rank, self.target_vec)
        plt.plot(alpha.numpy())
        plt.savefig('alpha.png')
        return alpha


def train_unit_transport(kernel_model, n_iters = 100000):
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr=kernel_model.params['learning_rate'])
    Loss_dict = {'n_iter': [], 'square_loss': []}
    kernel_model.train()
    for i in range(n_iters):
        loss, loss_dict = train_step(kernel_model, optimizer)
        if not i % kernel_model.params['print_freq']:
            print(f'At step {i}: square_loss = {round(float(loss_dict["square_loss"]),2)}')
            Loss_dict = update_list_dict(Loss_dict, loss_dict)
        #if not i % 10:
            #Y_pred = kernel_model.Z.detach().cpu().numpy() + kernel_model.X.detach().cpu().numpy()
            #sample_hmap(Y_pred,'../data/Y_in_progress.png', d = 1, range = (-3,3))

    return kernel_model, Loss_dict


def run():
    X = torch.tensor(unif_circle(1500))
    Y = torch.tensor(normal_proj_circle(1500))
    N = len(Y.T)
    params = { 'Y': Y,'print_freq': 1000, 'learning_rate':1, 'diff_map': geo_diffs}

    sample_hmap(Y.T, '../data/normal_proj_circle_good.png', d=2, bins=20)
    sample_hmap(X, '../data/unif_arc.png', d=2, bins=20)

    #W_rank = kernel_model.W_rank
    #e_vals = np.linalg.eig(W_rank)[0]
    #plt.plot(e_vals)
    #plt.plot(np.zeros(e_vals.shape), color='red')
    #plt.savefig('evals.png')
    #clear_plt()

    kernel_model = UnitKernel(params)
    W_rank = np.asarray(kernel_model.W_rank)
    plt.imshow(W_rank)
    plt.savefig('Wranks.png')
    clear_plt()

    target = np.asarray(kernel_model.target_vec)

    def f(x):
        y = (np.dot(W_rank, x) - target) ** 2
        #reg = 1e8 * (x.T @ W_inv @ x)
        return np.dot(y, y)

    x_0 = np.full(N, 1/N)

    bnds = [(0, np.inf) for i in range(N)]
    result = minimize(f, x_0,method='L-BFGS-B', bounds=bnds,
                      options={'disp': True, 'maxiter': 1000, 'maxfun': 500000, 'gtol': 0,'ftol': 1e-15})
    alpha = result['x']
    alpha = (alpha/np.linalg.norm(alpha,ord = 1)).reshape(len(alpha))

    plt.plot(alpha)
    plt.savefig('alpha.png')
    clear_plt()

    theta = get_theta(Y[0], Y[1])
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    alpha_sorted = alpha[sort_idx]

    plt.plot(theta_sorted, alpha_sorted)
    plt.savefig('alpha v theta.png')
    clear_plt()

    plt_vec = (W_rank @ alpha) / np.linalg.norm(W_rank @ alpha, ord = 1)
    plt.plot(plt_vec.reshape(len(alpha)))
    plt.savefig('Walpha.png')
    clear_plt()


    W = np.asarray(kernel_model.W)
    eps_vals = np.linspace(0, np.max(W), 40)
    W_eps = [(W <= eps).astype(np.float32) for eps in eps_vals]

    for W in W_eps:
       plt.plot((W @ alpha))

    plt.savefig('Weps_plots_big.png')
    clear_plt()


    resample_indexes = np.random.choice(np.arange(N), size=10000, replace = True, p = alpha)
    Y_resample = Y[:, resample_indexes]
    sample_hmap(Y_resample.T, '../data/resampled_arc.png', d=2, bins=20)
    clear_plt()






if __name__=='__main__':
    run()



