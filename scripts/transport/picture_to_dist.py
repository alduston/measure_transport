import numpy as np
import PIL
from PIL import Image,ImageOps
from unif_transport import resample
from get_data import sample_normal, sample_moons
import matplotlib.pyplot as plt


def array_to_sample(im_array, base_val = 255):
    W,L = im_array.shape
    R = W/L
    sample = []
    N = 0
    for iw in range(W):
        for il in range(L):
            n = abs(im_array[iw, il] - base_val)//25
            N += n
            x = -R + R*(2 * iw/W)
            y = -1 + (2 * il/L)
            loc = [[y,x]]
            sample += n * loc
    sample_array = np.asarray(sample).reshape(N,2)
    return sample_array


def process_img(img_name, gray = True, flip = True):
    img = Image.open(f'../../data/images/{img_name}.png')
    if gray:
        img = ImageOps.grayscale(img)
    if flip:
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    img_array = np.asarray(img)
    return img_array


def sample_elden_ring(N):
    img_array = process_img('elden_ring')
    img_base_sample = array_to_sample(img_array, base_val=0)
    sample = resample(img_base_sample.T, N=N)
    return sample


def run():
    N = 500000
    elden_sample = sample_elden_ring(N)
    plt_range = None
    save_loc = '../../data/images/elden_sample.png'
    #sample_hmap(elden_sample.T, save_loc,  d=2, bins=100, range=plt_range)

if __name__=='__main__':
    run()
