import numpy as np
import PIL
from PIL import Image,ImageOps
import matplotlib.pyplot as plt


def resample(Y, alpha = [], N = 10000):
    n = len(Y.T)
    if not len(alpha):
        alpha = np.full(n, 1/n)
    resample_indexes = np.random.choice(np.arange(n), size=N, replace=True, p=alpha)
    Y_resample = Y[:, resample_indexes]
    return Y_resample


def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return True


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


def array_to_sample(im_array, base_val = 255, p  = 1, M = 25):
    W,L = im_array.shape
    R = W/L
    sample = []
    N = 0
    for iw in range(W):
        for il in range(L):
            n = int((abs(im_array[iw, il] - base_val)**p)//M)
            N += n
            x = -R + R*(2 * iw/W)
            y = -1 + (2 * il/L)
            loc = [[y,x]]
            sample += n * loc
    sample_array = np.asarray(sample).reshape(N,2)
    return sample_array


def process_img(img_name, gray = True, flip = True, q = 0.0):
    img = Image.open(f'../../data/images/{img_name}.png')
    if gray:
        img = ImageOps.grayscale(img)
    if flip:
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    img_array = np.copy(np.asarray(img))
    b = np.quantile(img_array, q = q)
    img_array[img_array < b] *= 0
    return img_array


def sample_elden_ring(N):
    img_array = process_img('elden_ring')
    img_base_sample = array_to_sample(img_array, base_val=0)
    sample = resample(img_base_sample.T, N=N)
    return sample.T

def sample_twisted_rings(N):
    img_array = process_img('rings')
    img_base_sample = array_to_sample(img_array, 255)
    sample = resample(img_base_sample.T, N=N)
    return sample.T


def sample_bambdad(N, p = 2, M = 300):
    img_array = process_img('Bambdad', q= .15)
    #base_val = np.min(img_array)
    img_base_sample = array_to_sample(img_array, base_val=193, p = p, M = M)
    sample = resample(img_base_sample.T, N=N)
    return sample.T


def sample_boobs(N, p = 2, M = 300):
    img_array = process_img('boobs', q= .15)
    #base_val = np.min(img_array)
    img_base_sample = array_to_sample(img_array, p = p, M = M)
    sample = resample(img_base_sample.T, N=N)
    return sample.T


def run():
    N = 50000
    boob_sample = sample_boobs(N)
    plt_range = None
    save_loc = '../../data/images/boob_sample.png'
    sample_hmap(boob_sample, save_loc,  d=2, bins=100, range=plt_range)

if __name__=='__main__':
    run()
