from . import sampling

import os
from os.path import join

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import data
import model

def _test_loss(cinn, dataset, args, test_data=True):
    count = 0
    tot_loss = 0

    if test_data:
        loader = dataset.test_loader
    else:
        loader = [(dataset.val_x, dataset.val_y)]

    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            # the batch size is not the same for all test batches
            batch_size = x.shape[0]

            tot_loss += batch_size * cinn.nll(x, y).item()
            count += batch_size

    return tot_loss / count


def _average_batch_norm(cinn, dataset, args):

    cinn.train()
    bn_layers = []

    for module in cinn.inn.module_list:
        for subnet in module.children():
            for layer in subnet.children():
                if type(layer) == torch.nn.BatchNorm2d:
                    bn_layers.append(layer)

    bn_layers.append(cinn.cond_net.bn1)
    for levels in cinn.cond_net.layers.values():
        for block in levels.children():
            for layer in block.children():
                if type(layer) == torch.nn.BatchNorm2d:
                    bn_layers.append(layer)


    for l in bn_layers:
        l.reset_running_stats()
        l.momentum = None

    instance_count = len(bn_layers)

    assert instance_count > 0, "No batch norm layers found. Is the model constructed differently?"
    print('INSTANCES', instance_count)

    tot_iterations = 10000
    it = 0
    progress = tqdm(total=tot_iterations, ascii=True, ncols=100)

    with torch.no_grad():
        while it <= tot_iterations:
            for x, y in dataset.train_loader:
                x, y = x.cuda(), y.cuda()
                nll = cinn.nll(x, y)
                progress.update()

                if it >= tot_iterations:
                    progress.close()
                    cinn.eval()
                    return

                it += 1

def test(args):

    out_dir = args['checkpoints']['output_dir']
    figures_output_dir = join(out_dir, 'testing')
    os.makedirs(figures_output_dir, exist_ok=True)

    print('. Loading the dataset')
    dataset = data.MapToSatDataset(args)

    print('. Constructing the model')
    cinn = model.CINN(args)
    cinn.cuda()

    print('. Loading the checkpoint')
    try:
        cinn.load(join(out_dir, 'checkpoint_end_avg.pt'))
    except FileNotFoundError:
        print('. Averaging BatchNorm layers')
        cinn.load(join(out_dir, 'checkpoint_end.pt'))
        _average_batch_norm(cinn, dataset, args)
        cinn.save(join(out_dir, 'checkpoint_end_avg.pt'))
        pass

    cinn.eval()

    print('. Computing test loss')
    loss = _test_loss(cinn, dataset, args, test_data=True)
    print('TEST LOSS', loss)
    with open(join(figures_output_dir, 'test_loss'), 'w') as f:
        f.write(str(loss))

    print('. Generating samples')
    os.makedirs(join(figures_output_dir, 'samples'), exist_ok=True)
    for t in np.linspace(0.7, 1.2, num=20):
        sampling.sample(cinn, dataset, args, temperature=t, test_data=True, big_size=True)
        plt.savefig(join(figures_output_dir, 'samples', 'temp_{:.3f}.pdf'.format(t)))
        plt.close()

    print('. Visualizing feature pyramid')
    from .features_pca import features_pca
    features_pca(cinn, dataset, args, join(figures_output_dir, 'c_pca'))

def checkpoint_figures(path, model, dataset, args, test_data=False):
    sampling.sample(model, dataset, args, test_data)
    plt.savefig(path)
    plt.close()
