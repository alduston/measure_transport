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
#from kernel_geodesics import geo_diffs, boosted_geo_diffs
from copy import deepcopy
import random
from datetime import datetime as dt
from seaborn import kdeplot

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
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr= kernel_model.params['learning_rate'])
    Loss_dict = {'n_iter': [], 'fit': [], 'reg': [], 'test': [], 'total': []}
    kernel_model.train()
    for i in range(n_iter):
        loss, loss_dict = train_step(kernel_model, optimizer)
        Loss_dict = update_list_dict(Loss_dict, loss_dict)
        iter = kernel_model.iters
        if not iter % kernel_model.params['print_freq']:
            print_str = f'At step {iter}: fit_loss = {round(float((loss_dict["fit"])),6)},' + f' reg_loss = {round(float(loss_dict["reg"]),6)}'
            kernel_model.eval()
            test_loss = kernel_model.loss_test().detach().cpu()
            print_str += f', test loss = {round(float(test_loss),6)}'
            mem_str = ''
            if torch.cuda.is_available():
                free_mem, total_mem = torch.cuda.mem_get_info()
                mem_str = f', Using {round(100*(1-(free_mem/total_mem)),2)}% GPU mem'
            print(print_str + mem_str)
            kernel_model.train()
    return kernel_model, Loss_dict


def train_step(kernel_model, optimizer):
    optimizer.zero_grad()
    loss, loss_dict = kernel_model.loss()
    loss.backward()
    optimizer.step()

    kernel_model.iters += 1
    loss_dict['n_iter'] = kernel_model.iters
    return loss, loss_dict


def three_d_scatter(x,y,z, saveloc):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(x, y, z, color="green")
    plt.savefig(saveloc)
    return True


def seaborne_hmap(sample, save_loc,  d = 2, range = None, scmap = 'Blues'):
    try:
        sample = sample.detach().cpu()
    except AttributeError:
        pass
    if d == 2:
        x, y = sample.T
        x = np.asarray(x)
        y = np.asarray(y)

        kdeplot(x=x, y=y, fill=True, bw_adjust=0.25, cmap=scmap)
        plt.xlim(range[0][0], range[0][1])
        plt.ylim(range[1][0], range[1][1])

    elif d == 1:
        kdeplot(data=x, bw_adjust=0.25)
        plt.xlim(range[0][0], range[0][1])
    plt.savefig(save_loc)
    clear_plt()
    return True




def sample_hmap(sample, save_loc, bins = 20, d = 2, range = None, vmax= None,
                cmap = None, scmap = 'Blues', bw_adjust=0.25):
    try:
        sample = sample.detach().cpu()
    except AttributeError:
        pass
    plt.figure(figsize=(10, 4))
    if d == 2:
        plt.subplot(1, 2, 1)
        x, y = sample.T
        x = np.asarray(x)
        y = np.asarray(y)
        plt.hist2d(x,y, density=True, bins = bins, range = range, cmin = 0, vmin=0, vmax = vmax, cmap = cmap)
        plt.colorbar()

        plt.subplot(1, 2, 2)
        kdeplot(x=x, y=y, fill=True,bw_adjust=bw_adjust, cmap=scmap)
        plt.xlim(range[0][0],range[0][1])
        plt.ylim(range[1][0], range[1][1])

    elif d == 1:
        plt.subplot(1, 2, 1)
        x =  sample
        x = np.asarray(x)
        plt.hist(x, bins = bins, range = range)

        plt.subplot(1, 2, 2)
        kdeplot(data = x, fill=True, bw_adjust=bw_adjust)
        plt.xlim(range[0], range[0])
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
    size = 6
    s = [size for x in x]
    plt.scatter(x,y, s=s)

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


def run():
   pass



if __name__=='__main__':
    run()
