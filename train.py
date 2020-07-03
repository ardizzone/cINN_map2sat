import os
from os.path import join
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import data
import model
from evaluation import checkpoint_figures

def _check_gradients_per_block(inn):
    print('===' * 10)
    for i, mod in enumerate(inn.module_list):
        grad_norm_list = []
        for p in mod.parameters():
            if p.grad is not None:
                grad_norm_list.append(1000. * torch.norm(p.grad.data**2).item())

        if grad_norm_list:
            mean = '{:.5f}'.format(sum(grad_norm_list) / len(grad_norm_list))
        else:
            mean = '--'

        print('{:>4d}  {:>10s}'.format(i, mean))
    print('===' * 10)


def train(args):

    cinn = model.CINN(args)
    cinn.train()
    cinn.cuda()
    n_gpus = eval(args['training']['parallel_GPUs'])
    if n_gpus > 1:
        cinn_parallel = nn.DataParallel(cinn, list(range(n_gpus)))
    else:
        cinn_parallel = cinn

    dataset = data.MapToSatDataset(args)

    log_interval         = 1 #print losses every epoch
    checkpoint_interval  = eval(args['checkpoints']['checkpoint_interval'])
    checkpoint_overwrite = eval(args['checkpoints']['checkpoint_overwrite'])
    checkpoint_on_error  = eval(args['checkpoints']['checkpoint_on_error'])
    figures_interval     = eval(args['checkpoints']['figures_interval'])
    figures_overwrite    = eval(args['checkpoints']['figures_overwrite'])
    no_progress_bar      = not eval(args['checkpoints']['epoch_progress_bar'])

    N_epochs = eval(args['training']['N_epochs'])
    output_dir = args['checkpoints']['output_dir']

    checkpoints_dir = join(output_dir, 'checkpoints')
    figures_dir = join(output_dir, 'figures')

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    logfile = open(join(output_dir, 'losses.dat'), 'w')

    def log_write(string):
        logfile.write(string + '\n')
        print(string, flush=True)


    log_header = '{:>8s}{:>10s}{:>12s}{:>12s}'.format('Epoch', 'Time (m)', 'NLL train', 'NLL val')
    log_fmt    = '{:>8d}{:>10.1f}{:>12.5f}{:>12.5f}'

    log_write(log_header)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, gamma=0.1,
                                 milestones=eval(args['training']['milestones_lr_decay']))

    val_x = dataset.val_x.cuda()
    val_y = dataset.val_y.cuda()

    if figures_interval > 0:
        checkpoint_figures(join(figures_dir, 'init.pdf'), cinn, dataset, args)

    t_start = time.time()

    for epoch in range(N_epochs):
        progress_bar = tqdm(total=dataset.epoch_length, ascii=True, ncols=100, leave=False,
                            disable=no_progress_bar)

        loss_per_batch = []

        for x, y in dataset.train_loader:
            x, y = x.cuda(), y.cuda()

            nll = cinn_parallel(x, y).mean()
            nll.backward()
            # _check_gradients_per_block(cinn.inn)
            loss_per_batch.append(nll.item())

            cinn.optimizer.step()
            cinn.optimizer.zero_grad()
            progress_bar.update()

        # end of epoch:
        scheduler.step()
        progress_bar.close()

        if (epoch + 1) % log_interval == 0:

            with torch.no_grad():
                time_delta = (time.time() - t_start) / 60.
                train_loss = np.mean(loss_per_batch)
                val_loss = cinn_parallel(val_x, val_y).mean()

            log_write(log_fmt.format(epoch + 1, time_delta, train_loss, val_loss))

        if figures_interval > 0 and (epoch + 1) % figures_interval == 0:
            checkpoint_figures(join(figures_dir, 'epoch_{:05d}.pdf'.format(epoch + 1)), cinn, dataset, args)

        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            cinn.save(join(checkpoints_dir, 'checkpoint_{:05d}.pt'.format(epoch + 1)))

    logfile.close()
    cinn.save(join(output_dir, 'checkpoint_end.pt'))
