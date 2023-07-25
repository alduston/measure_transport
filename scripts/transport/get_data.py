import numpy as np
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
from ellipse import rand_ellipse
from scipy.stats import norm


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)

    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        return inf_train_gen("8gaussians", rng, batch_size)


def resample(Y, alpha = [], N = 10000):
    n = len(Y.T)
    if not len(alpha):
        alpha = np.full(n, 1/n)
    resample_indexes = np.random.choice(np.arange(n), size=N, replace=True, p=alpha)
    Y_resample = Y[:, resample_indexes]
    return Y_resample


def sample_normal(N = 100, d = 2):
    mu = np.zeros(d)
    sigma = np.identity(d)
    X_sample = np.random.multivariate_normal(mu, sigma, N)
    X_sample = X_sample.reshape(N,d)
    return X_sample

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


def cut_bool(v, thresh = .85):
    return np.min(np.abs(v)) < thresh


def cut_square(N = 200):
    square = unif_square(3 * N)
    cut_square = np.asarray([v for v in square if cut_bool(v)])
    return cut_square[:N, :]


def unif_circle(N = 200):
    theta = np.random.uniform(low = -np.pi, high = np.pi, size = N)
    X = np.cos(theta)
    Y = np.sin(theta)
    sample = np.asarray([[x,y] for x,y in zip(X,Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample


def unif_elipse(N = 1000, a = .4, b = 3):
    X, Y = rand_ellipse(a = a, b = b, size=N)
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    return sample


def two_unif_circle(N = 200):
    theta = np.random.uniform(low = -np.pi, high = np.pi, size = N)
    X = np.cos(theta)
    Y = np.sin(theta)

    Y[:N//2] += 2
    Y[:N // 2] -= 2
    sample = np.asarray([[x,y] for x,y in zip(X,Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample


def sample_unif_dumbell(N =200):
    n = N//2
    s_1 = unif_square_2d(N = n, x_range = [-3,-1], y_range = [-.6,.6])
    s_2 = unif_square_2d(N = n//8, x_range=[-1, 1], y_range=[-.015, .015])
    s_3 = unif_square_2d(N = n, x_range=[1, 2], y_range=[-.3, .3])

    dumbell_sample = np.concatenate([s_1, s_2, s_3])
    return dumbell_sample


def one_normalize(vec):
    return vec/np.linalg.norm(vec, ord = 1)


def resample(Y, alpha = [], N = 10000):
    n = len(Y.T)
    if not len(alpha):
        alpha = np.full(n, 1/n)
    resample_indexes = np.random.choice(np.arange(n), size=N, replace=True, p=alpha)
    Y_resample = Y[:, resample_indexes]
    return Y_resample


def normal_theta_circle(N = 500):
    thetas = np.linspace(-np.pi, np.pi, N)
    theta_probs = one_normalize(np.exp(-2 * np.abs(thetas)) + .01)
    thetas = thetas.reshape((1,len(thetas)))
    thetas = resample(thetas, theta_probs, N).reshape(thetas.shape[1])
    X = np.cos(thetas)
    Y = np.sin(thetas)
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample.T


def normal_theta_circle_noisy(N = 500, scale = 5e-2):
    thetas = np.linspace(-np.pi, np.pi, N)
    theta_probs = one_normalize(np.exp(-2 * np.abs(thetas)) + .01)
    thetas = thetas.reshape((1,len(thetas)))
    thetas = resample(thetas, theta_probs, N).reshape(thetas.shape[1])
    X = np.cos(thetas)
    X +=  np.random.randn(*X.shape) * scale
    Y = np.sin(thetas)
    Y += np.random.randn(*Y.shape) * scale
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample.T


def normal_theta_two_circle(N = 500):
    thetas = np.linspace(-np.pi, np.pi, N)
    theta_probs = one_normalize(np.exp(-2 * np.abs(thetas)) + .01)
    thetas = thetas.reshape((1,len(thetas)))
    thetas = resample(thetas, theta_probs, N).reshape(thetas.shape[1])
    X = np.cos(thetas)
    Y = np.sin(thetas)
    Y[:N // 2] += 2
    Y[N // 2:] -= 2

    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample.T


def sample_2normal(N = 100, d = 2, mu_1 = 1, mu_2 = -1, sigma = .5):
    mu_1 = mu_1 * np.ones(d)
    mu_2 =  mu_2  * np.ones(d)
    sigma =  sigma * np.identity(d)
    X_1 = np.random.multivariate_normal(mu_1, sigma, N)
    X_2 = np.random.multivariate_normal(mu_2, sigma, N)
    X_sample =  np.concatenate([X_1, X_2]).reshape(2*N)
    return X_sample


def sample_uniform(N = 100,  d = 2, l = -1.5, h = 1.5):
    Y = []
    for i in range(d):
        yi = np.random.uniform(l,h, N)
        Y.append(yi)
    Y_sample = np.stack(Y).reshape((N,d))
    return Y_sample


def sample_banana(N):
    xx = np.random.randn(1, N)
    zz = np.random.randn(1, N)
    Y = np.concatenate((xx, np.power(xx, 2) + 0.3 * zz), 1).reshape(N, 2)
    return Y

def proj_circle(N = 500):
    X = np.random.uniform(low=-1, high=1, size=N)
    Y = np.sqrt(np.ones(len(X)) - X**2) * np.random.choice([-1.0,1.0], size=N)
    sample = np.stack([X,Y]).reshape((2, len(X)))
    return sample


def normal_theta_circle(N = 500):
    thetas = np.linspace(-np.pi, np.pi, N)
    theta_probs = one_normalize(np.exp(-2 * np.abs(thetas)) + .02)
    thetas = thetas.reshape((1,len(thetas)))
    thetas = resample(thetas, theta_probs, N).reshape(thetas.shape[1])
    X = np.cos(thetas)
    Y = np.sin(thetas)
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample.T


def normal_proj_circle( N = 500):
    X = np.random.normal(loc=0, scale=1, size=10*N)
    X = X[np.abs(X) <= 1][:N]
    Y = np.sqrt(np.ones(len(X)) - X**2) * np.random.choice([-1.0,1.0], size=N)
    sample = np.stack([X,Y]).reshape((2, len(X)))
    return sample


def two_normal_proj_circle( N = 500):
    X = sample_2normal(N = 10 * N, mu_1 = .8, mu_2 = -.8, sigma = .15, d=1)
    X = X[np.abs(X) <= 1][:N]
    Y = np.sqrt(np.ones(len(X)) - X**2) * np.random.choice([-1.0,1.0], size=N)
    sample = np.stack([X,Y]).reshape((2, len(X)))
    return sample


def sin_proj_circle( N = 500):
    theta = np.random.uniform(low=-2 * np.pi, high=2 *np.pi, size=N)
    weights = np.abs(np.sin(theta))
    weights = weights/np.sum(weights)
    theta =  np.random.choice(theta, size=N, replace=True, p=weights)
    X = np.cos(theta)
    Y = np.sin(theta)
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    X, Y = sample.T[0], sample.T[1]
    return sample.T


def unif_square_2d(N = 200, x_range = [-1,1], y_range = [-1,1]):
    X = np.random.uniform(low = x_range[0], high = x_range[1], size = N)
    Y = np.random.uniform(low = y_range[0], high = y_range[1], size = N)
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    return sample


def mgan1(N = 200, x_range = [-3,3], eps_var = .05):
    Y = np.random.uniform(low = x_range[0], high = x_range[1], size = N)
    eps = np.random.gamma(shape = 1, scale=.3, size=N)
    U = np.tanh(Y) + eps
    sample = np.asarray([[y, u] for y, u in zip(Y, U)])
    return sample


def mgan2(N = 200, x_range = [-3,3], eps_var = .05**.5):
    Y = np.random.uniform(low = x_range[0], high = x_range[1], size = N)
    eps = np.random.normal(loc=0, scale=eps_var, size=N)
    U = np.tanh(Y + eps)
    sample = np.asarray([[y, u] for y, u in zip(Y, U)])
    return sample


def mgan3(N = 200, x_range = [-3,3], eps_var = .05):
    Y = np.random.uniform(low = x_range[0], high = x_range[1], size = N)
    eps = np.random.gamma(shape = 1, scale=.3, size=N)
    U = eps * np.tanh(Y)
    sample = np.asarray([[y, u] for y, u in zip(Y, U)])
    return sample


def norm_square_2d(N = 200):
    X = np.random.uniform(low = -1, high = 1, size = N)
    X = X * np.abs(X)
    Y = np.random.uniform(low = -1, high = 1, size = N)
    Y = Y * np.abs(Y)
    sample = np.asarray([[x, y] for x, y in zip(X, Y)])
    return sample


def sample_swiss_roll(N):
    return inf_train_gen("swissroll", batch_size=N)


def sample_moons(N):
    return inf_train_gen("moons", batch_size=N)


def sample_rings(N):
    return inf_train_gen("rings", batch_size=N)

def sample_circles(N):
    return inf_train_gen("circles", batch_size=N)


def sample_spirals(N):
    return inf_train_gen("2spirals", batch_size=N)


def sample_pinweel(N):
    return inf_train_gen("pinwheel", batch_size=N)

def geq_1d(tensor):
    if not len(tensor.shape):
        tensor = tensor.reshape(1,1)
    elif len(tensor.shape) == 1:
        tensor = tensor.reshape(len(tensor), 1)
    return tensor

def normal_density(X):
    return np.exp(-np.linalg.norm(geq_1d(X), axis = 0)**2)


def KL(sample, ref_sample = [], ref_density = normal_density):
    sample_densitys = np.asarray([ref_density(x) for x in sample])
    KL_div = np.sum(np.log(1/ sample_densitys))
    return KL_div


def banana_density(X):
    pass



def run():
    data = inf_train_gen("swissroll")



if __name__=='__main__':
    run()