from . import sampling

import os
from os.path import join

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import data
import model

def _test_loss(cinn, dataset, args):
    count = 0
    tot_loss = 0
    with torch.no_grad():
        for x, y in dataset.test_loader:
            x, y = x.cuda(), y.cuda()
            # the batch size is not the same for all test batches
            batch_size = x.shape[0]

            tot_loss += batch_size * cinn.nll(x, y).item()
            count += batch_size

    return tot_loss / count


def _average_batch_norm(cinn, dataset, args):

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

    tot_iterations = 2000
    it = 0
    progress = tqdm(total=tot_iterations, ascii=True, ncols=100)

    cinn.train()
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

    dataset = data.MapToSatDataset(args)

    cinn = model.CINN(args)
    cinn.cuda()

    try:
        cinn.load(join(out_dir, 'checkpoint_end_avg.pt'))
    except FileNotFoundError:
        cinn.load(join(out_dir, 'checkpoint_end.pt'))
        _average_batch_norm(cinn, dataset, args)
        cinn.save(join(out_dir, 'checkpoint_end_avg.pt'))
        pass
    cinn.eval()

    print('TEST LOSS', _test_loss(cinn, dataset, args))
    checkpoint_figures(join(figures_output_dir, 'samples.pdf'),
                       cinn, dataset, args, test_data=True)

def checkpoint_figures(path, model, dataset, args, test_data=False):
    sampling.sample(model, dataset, args, test_data)
    plt.savefig(path)
    plt.close()
