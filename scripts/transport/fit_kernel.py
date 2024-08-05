import torch
import numpy as np
import matplotlib.pyplot as plt
#import cv2
import torch.nn as nn
import os
import shutil

#from kernel_geodesics import geo_diffs, boosted_geo_diffs
from seaborn import kdeplot

import warnings
warnings.filterwarnings("ignore")


def format(n, n_digits = 6):
    try:
        if n > 1e-3:
            return round(n,n_digits)
        a = '%E' % n
        str =  a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
        scale = str[-4:]
        digits = str[:-4]
        return digits[:min(len(digits),n_digits)] + scale
    except IndexError:
        return n

def clear_plt():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    #plt.figure(plt.rcarams.get('figure.figsize'))
    return True


def update_list_dict(Dict, update):
    for key, val in update.items():
        Dict[key].append(val)
    return Dict


def prob_normalization(alpha):
    N = len(alpha)
    c_norm = torch.log(N / torch.sum(torch.exp(alpha)))
    return alpha - c_norm


def print_losses(loss_dict):
    print_str = f'At step {loss_dict["n_iter"]}: '
    for key in loss_dict.keys():
        if key in ['fit', 'reg', 'inv']:
            print_str +=  f'{key}_loss = {format(float((loss_dict[key])))}, '
    print_str += f'test mmd = {format(float(loss_dict["test_mmd"]))}, '
    print_str += f'test emd = {format(float(loss_dict["test_emd"]))}, '
    print_str += f'grad norm = {format(float(loss_dict["grad_norm"]))}'
    mem_str = ''
    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        mem_str = f', Using {round(100 * (1 - (free_mem / total_mem)), 2)}% GPU mem'
    print_str += mem_str
    #print(print_str)



def train_kernel(kernel_model, n_iter = np.inf):
    optimizer = torch.optim.Adam(kernel_model.parameters(), lr= kernel_model.params['learning_rate'])
    kernel_model.eval()
    Loss_dict = {key: [val] for key,val in kernel_model.loss()[1].items()}
    Loss_dict['n_iter'] = [0]
    test_mmd, test_emd = kernel_model.loss_test()
    Loss_dict['test_mmd'] = [test_mmd.detach().cpu()]
    Loss_dict['test_emd'] = [test_emd]
    Loss_dict['grad_norm'] = [0]
    kernel_model.train()
    iter = kernel_model.iters
    grad_norm = np.inf
    i = 0
    while grad_norm > kernel_model.params['grad_cutoff'] and i < n_iter:
        if not iter % kernel_model.params['print_freq']:
            test_mmd, test_emd = kernel_model.loss_test()
            test_mmd = test_mmd.detach().cpu()
        kernel_model.train()
        loss, loss_dict = train_step(kernel_model, optimizer)
        if not iter % kernel_model.params['print_freq'] or not i:
            grad_norm = kernel_model.total_grad()
            kernel_model.eval()
            loss_dict['test_mmd'] = test_mmd
            loss_dict['test_emd'] = test_emd
            loss_dict['n_iter'] = iter
            loss_dict['grad_norm'] = grad_norm
            Loss_dict = update_list_dict(Loss_dict, loss_dict)
            print_losses(loss_dict)
        iter = kernel_model.iters
        i += 1
    return kernel_model, Loss_dict


def train_step(kernel_model, optimizer):
    optimizer.zero_grad()
    loss, loss_dict = kernel_model.loss()
    loss.backward()
    nn.utils.clip_grad_norm_(kernel_model.parameters(), 1000)
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




def sample_hmap(sample, save_loc, bins = 70, d = 2, range = None, vmax= None,
                cmap = None, scmap = 'Blues', bw_adjust=0.25, cbar = True):

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
        plt.hist2d(x,y, density=True, bins = bins, range = range, cmin = 0, vmin=0, vmax = vmax,
                   cmap = cmap)
        if cbar:
            plt.colorbar()
        plt.subplot(1, 2, 2)
        try:
            kdeplot(x=x, y=y, fill=True,bw_adjust=bw_adjust, cmap=scmap)
            if range != None:
                plt.xlim(range[0][0],range[0][1])
                plt.ylim(range[1][0], range[1][1])
        except ValueError:
            pass

    elif d == 1:
        plt.subplot(1, 2, 1)
        x =  sample
        x = np.asarray(x)
        plt.hist(x, bins = bins, range = range)

        plt.subplot(1, 2, 2)
        try:
            kdeplot(data = x, fill=True, bw_adjust=bw_adjust)
            if range != None:
                plt.xlim(range[0], range[1])
        except ValueError:
            pass

    save_loc = save_loc.replace('//', '/')
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


def add_base_frame(save_loc, n = 7):
    frames = os.listdir(save_loc)
    for frame_name in frames:
        if frame_name.startswith('new_frame'):
            frame_num = frame_name[9:-4]
            new_frame_name = f'frame{int(frame_num) + n}.png'
            os.rename(f'{save_loc}/{frame_name}', f'{save_loc}/{new_frame_name}')
    base_frame_loc = f'{save_loc}/frame{n}.png'
    for i in range(n):
        target_loc = f'{save_loc}/frame{i}.png'
        shutil.copyfile(base_frame_loc, target_loc)
    return True


def copy_frames(save_loc, k = 3):
    frames = os.listdir(save_loc)
    frame_dir = f'{save_loc}/diff_frames'
    try:
        os.mkdir(frame_dir)
    except OSError:
        pass
    for frame_name in frames:
        if frame_name.startswith('frame'):
            frame_num = frame_name[5:-4]
            for i in range(k):
                new_frame_name = f'new_frame{(k * int(frame_num)) + i}.png'
                shutil.copyfile(f'{save_loc}/{frame_name}', f'{frame_dir}/{new_frame_name}')
            os.remove(f'{save_loc}/{frame_name}')
    return True


def dict_to_np(dict):
    for key,val in dict.items():
        try:
            dict[key] = val.detach().cpu().numpy()
        except BaseException:
            pass
    return dict


def process_frames(save_loc, n = 12, k = 3):
    copy_frames(save_loc, k = k)
    frame_dir = f'{save_loc}/diff_frames'
    add_base_frame(frame_dir, n = n)
    return True


def run():
    save_loc = '../../data/transport/mgan1_movie'
    process_frames(save_loc, k = 3)



if __name__=='__main__':
    run()
