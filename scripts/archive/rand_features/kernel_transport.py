import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from jax.config import config
from jax.scipy.optimize import minimize

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import torch
from ellipse import rand_ellipse


def one_normalize(vec):
    return vec/np.linalg.norm(vec, ord = 1)


def resample(Y, alpha=[], N=10000):
    n = len(Y.T)
    if not len(alpha):
        alpha = np.full(n, 1 / n)
    resample_indexes = np.random.choice(np.arange(n), size=N, replace=True, p=alpha)
    Y_resample = Y[:, resample_indexes]
    return Y_resample

def unif_circle(N = 200):
    theta = np.random.uniform(low = -np.pi, high = np.pi, size = N)
    X = np.cos(theta)
    Y = np.sin(theta)
    sample = np.asarray([[x,y] for x,y in zip(X,Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample


def unif_elipse(N = 1000, a = .5, b = 2):
    X, Y = rand_ellipse(a = a, b = b, size=N)
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    return sample

def unif_square(N = 200):
    o1_vals = np.random.choice([-1.0, 1.0], size = N)
    unifs_vals = np.random.uniform(low = -1, high = 1, size = N)
    o1_indexes = np.random.choice([0, 1], size = N)
    unif_indexes = np.abs(1 - o1_indexes).astype(int)
    samples = np.zeros((N,2))
    for i, sample in enumerate(samples):
        samples[i,o1_indexes[i]] = o1_vals[i]
        samples[i, unif_indexes[i]] = unifs_vals[i]
    return samples



def sample_normal(N = 100, d = 2):
    mu = np.zeros(d)
    sigma = np.identity(d)
    X_sample = np.random.multivariate_normal(mu, sigma, N)
    X_sample = X_sample.reshape(N,d)
    return X_sample

def sample_uniform(N = 100,  d = 2, l = -1.5, h = 1.5):
    Y = []
    for i in range(d):
        yi = np.random.uniform(l,h, N)
        Y.append(yi)
    Y_sample = np.stack(Y).reshape((N,d))
    return Y_sample


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True

def normalize(vec):
    normal_vec = vec - np.mean(vec, axis=0)
    normal_vec = normal_vec/np.var(normal_vec,axis = 0)
    return normal_vec


def l_scale(X, q = .25):
    return np.quantile(diff_matrix(X,X), q = q)


def diff_matrix(X,X_tilde):
    return jnp.linalg.norm(jnp.expand_dims(X, 1) - X_tilde, axis=2)


def radial_kernel(X, X_tilde, kern_params):
    norm_diffs = diff_matrix(X, X_tilde)
    sigma = kern_params['sigma']
    l = kern_params['l']
    res =  sigma * jnp.exp(-((norm_diffs/np.sqrt(2)*l) ** 2))
    return res


def linear_kernel(X, X_tilde, kern_params):
    sig_b = kern_params['sig_b']
    sig_v = kern_params['sig_v']
    c = kern_params['c']
    return sig_b**2 + (sig_v**2)*jnp.matmul(X-c, (X_tilde-c).T)


def poly_kernel(X, X_tilde, kern_params):
    c = kern_params['c']
    alpha = kern_params['alpha']
    return (c + jnp.matmul(X, X_tilde.T))**alpha


def get_kernel(kernel_params):
    kernel_name = kernel_params['name']

    if kernel_name == 'radial':
        return lambda x,x_tilde: radial_kernel(x,x_tilde, kernel_params)

    elif kernel_name == 'poly':
        return lambda x, x_tilde: poly_kernel(x, x_tilde, kernel_params)

    elif  kernel_name == 'linear':
        return lambda x, x_tilde: linear_kernel(x, x_tilde, kernel_params)


def map(z, alpha, X, kernel):
    k_Xz = kernel(X,z)
    return alpha.T @ k_Xz


def normalize_cols(W):
    return np.diag(np.diag(W @ W.T)**-1) @ W


def s_map(X_tidle, Z, X, kernel):
    k_Xz_n = normalize_cols(kernel(X, X_tidle))
    return k_Xz_n @ Z



def phi(z, bs, ws, kappa):
    return kappa * jnp.cos(ws @ z + bs)

def phi_vec(Z, bs, ws, kappa):
    return jnp.asarray([phi(z, bs, ws, kappa) for z in Z])


def MMD(Z, Y, bs, ws):
    N_features = 100
    kappa = np.sqrt(2 / N_features)
    return jnp.linalg.norm(jnp.mean(phi_vec(Z,bs, ws, kappa), axis=0) - jnp.mean(phi_vec(Y, bs, ws, kappa), axis=0))


def exact_mmd(Z, Y, mmd_kernel,N):
    k_ZZ = mmd_kernel(Z,Z)
    k_ZZ = k_ZZ - jnp.diag(jnp.diag(k_ZZ))
    k_ZY = mmd_kernel(Z, Y)

    normal_factor = N / (N - 1)
    return normal_factor * (jnp.mean(k_ZZ)) - 2 * jnp.mean(k_ZY)


def sample_hist(sample, save_loc, bins = 20, range = None):
    x =  sample
    x = np.asarray(x)
    plt.hist(x, bins = bins, range = range)
    plt.savefig(save_loc)
    clear_plt()
    return True


def sample_scatter(sample, save_loc, bins=20, d=2, range=[]):
    x, y = sample.T
    x = np.asarray(x)
    y = np.asarray(y)
    plt.scatter(x, y)

    if len(range):
        x_left, x_right = range[0]
        y_bottom, y_top = range[1]
        plt.xlim(x_left, x_right)
        plt.ylim(y_bottom, y_top)
    plt.savefig(save_loc)
    clear_plt()
    return True


def run():
    exp_name = 'banana'
    save_dir = f'../../data/kernel_transport/{exp_name}'
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    N = 500
    plt_range = [[-4, 4], [-4, 4]]
    nugget = 1e-5
    reg_lambda = 1e-5
    N_features = 100

    kappa = np.sqrt(2 / N_features)

    X = unif_circle(N)
    X = normalize(np.random.randn(N, 2))
    l = l_scale(X)

    bs = np.random.uniform(low=0, high=2 * np.pi, size=N_features)
    ws = np.random.multivariate_normal(mean=np.zeros(2), cov=(l**2)*np.eye(2), size=N_features)


    def phi(x):
        return kappa * jnp.cos(ws @ x + bs)

    phi_vec = jit(vmap(phi))

    def feature_mmd(A, B):
        return jnp.linalg.norm(jnp.mean(phi_vec(A), axis=0) - jnp.mean(phi_vec(B), axis=0))


    def diff_test(A,B, mmd_kernel, l):
        k_AB = mmd_kernel(A, B)
        x = A[0]
        phi_x = phi(x)

        vals = np.linspace(np.min(k_AB), np.max(k_AB), 200)
        k_vals = []
        feat_k_vals = []
        for val in vals:
            k_val = np.exp(-((val**2)/(2 *(l**2))))
            c = np.full(2, val/np.sqrt(2))
            y = x + c
            phi_y = phi(y)

            feat_k_val = jnp.inner(phi_x, phi_y)

            k_vals.append(k_val)
            feat_k_vals.append(feat_k_val)

        feat_k_vals = np.asarray(feat_k_vals).reshape(vals.shape)
        k_vals = np.asarray(k_vals).reshape(vals.shape)

        plt.plot(vals, feat_k_vals, label='Rand features')
        plt.plot(vals, k_vals, label='True')
        plt.legend()
        plt.savefig('random_feature_approx_errors.png')
        return True


    #X = unif_elipse(N, .5, 2)

    xx = np.random.randn(1, N)
    zz = np.random.randn(1, N)
    Y = normalize(np.concatenate((xx, np.power(xx, 2) + 0.3 * zz), 1).reshape(2, N).T)
    #Y = unif_circle(N)

    sample_scatter(X, f'{save_dir}/Xtrain.png', range =  plt_range)
    sample_scatter(Y, f'{save_dir}/Ytrain.png', range=plt_range)

    fit_kernel_params = {'name': 'radial', 'l': l, 'sigma': 1}
    mmd_kernel_params = {'name': 'radial', 'l': l, 'sigma': 1}

    fit_kernel = get_kernel(fit_kernel_params)
    mmd_kernel = get_kernel(fit_kernel_params)

    #diff_test(X,Y,mmd_kernel, l)

    k_XX = fit_kernel(X,X)
    k_XX_inv = np.linalg.inv(k_XX + nugget*np.identity(N))


    @jit
    def objective_f(z):
        Z = z.reshape(N, 2)
        fit_loss = feature_mmd(Z + X, Y)
        reg_loss = jnp.trace(Z.T @ k_XX_inv @ Z)
        res = fit_loss + reg_loss * reg_lambda
        return res

    z_0 = jnp.zeros(N*2)
    res = minimize(objective_f, z_0, method='BFGS', tol=1e-3)
    Z = res.x
    Z = Z.reshape(N,2)

    X_2 = normalize(np.random.randn(N, 2))
    alpha = k_XX_inv @ Z
    Y_2 = map(X_2, alpha, X, fit_kernel).reshape(N, 2)

    l_smooth = l_scale(Y)/(2*l_scale(X))
    smooth_kernel_params = {'name': 'radial', 'l': l_smooth, 'sigma': 1}
    smooth_kernel = get_kernel(fit_kernel_params)

    Z_map= s_map(X, Z, X, smooth_kernel).reshape(N,2)
    Z_2 = s_map(X, Z, X_2, smooth_kernel).reshape(N, 2)

    plt.scatter(*X.T, label='reference')
    plt.scatter(*(X + Z_map).T, label='mapped')
    plt.scatter(*Y.T, label='target')
    plt.scatter(*(X_2 + Z_2).T, label='mapped new')
    plt.legend()
    plt.savefig(f'{save_dir}/Ypred.png')



if __name__=='__main__':
    run()