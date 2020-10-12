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

    ##########################
    # Relevant config values #
    ##########################

    log_interval         = 1 #print losses every epoch
    checkpoint_interval  = eval(args['checkpoints']['checkpoint_interval'])
    checkpoint_overwrite = eval(args['checkpoints']['checkpoint_overwrite'])
    checkpoint_on_error  = eval(args['checkpoints']['checkpoint_on_error'])
    figures_interval     = eval(args['checkpoints']['figures_interval'])
    figures_overwrite    = eval(args['checkpoints']['figures_overwrite'])
    no_progress_bar      = not eval(args['checkpoints']['epoch_progress_bar'])

    N_epochs             = eval(args['training']['N_epochs'])
    output_dir           = args['checkpoints']['output_dir']
    n_gpus               = eval(args['training']['parallel_GPUs'])
    checkpoint_resume    = args['checkpoints']['resume_checkpoint']
    cond_net_resume      = args['checkpoints']['resume_cond_net']

    checkpoints_dir      = join(output_dir, 'checkpoints')
    figures_dir          = join(output_dir, 'figures')

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    #######################################
    # Construct and load network and data #
    #######################################

    cinn = model.CINN(args)
    cinn.train()
    cinn.cuda()

    if checkpoint_resume:
        cinn.load(checkpoint_resume)

    if cond_net_resume:
        cinn.load_cond_net(cond_net_resume)

    if n_gpus > 1:
        cinn_parallel = nn.DataParallel(cinn, list(range(n_gpus)))
    else:
        cinn_parallel = cinn

    scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, gamma=0.1,
                                 milestones=eval(args['training']['milestones_lr_decay']))

    dataset = data.dataset(args)
    val_x = dataset.val_x.cuda()
    val_y = dataset.val_y.cuda()

    x_std, y_std = [], []
    x_mean, y_mean = [], []

    with torch.no_grad():
        for x, y in tqdm(dataset.train_loader):
            x_std.append(torch.std(x, dim=(0,2,3)).numpy())
            y_std.append(torch.std(y, dim=(0,2,3)).numpy())
            x_mean.append(torch.mean(x, dim=(0,2,3)).numpy())
            y_mean.append(torch.mean(y, dim=(0,2,3)).numpy())
            break

    print(np.mean(x_std, axis=0))
    print(np.mean(x_mean, axis=0))

    print(np.mean(y_std, axis=0))
    print(np.mean(y_mean, axis=0))



    ####################
    # Logging business #
    ####################

    logfile = open(join(output_dir, 'losses.dat'), 'w')

    def log_write(string):
        logfile.write(string + '\n')
        logfile.flush()
        print(string, flush=True)

    log_header = '{:>8s}{:>10s}{:>12s}{:>12s}'.format('Epoch', 'Time (m)', 'NLL train', 'NLL val')
    log_fmt    = '{:>8d}{:>10.1f}{:>12.5f}{:>12.5f}'

    log_write(log_header)

    if figures_interval > 0:
        checkpoint_figures(join(figures_dir, 'init.pdf'), cinn, dataset, args)

    t_start = time.time()

    ####################
    #  V  Training  V  #
    ####################

    for epoch in range(N_epochs):
        progress_bar = tqdm(total=dataset.epoch_length, ascii=True, ncols=100, leave=False,
                            disable=True)#no_progress_bar)

        loss_per_batch = []

        for i, (x, y) in enumerate(dataset.train_loader):
            x, y = x.cuda(), y.cuda()

            nll = cinn_parallel(x, y).mean()
            nll.backward()
            # _check_gradients_per_block(cinn.inn)
            loss_per_batch.append(nll.item())
            print('{:03d}/445  {:.6f}'.format(i, loss_per_batch[-1]), end='\r')

            cinn.optimizer.step()
            cinn.optimizer.zero_grad()
            progress_bar.update()

        # from here: end of epoch
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
