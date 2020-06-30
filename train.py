import os
import time

from tqdm import tqdm
import numpy as np

import data
import model

def train(args):

    cinn = model.CINN(args)
    cinn.train()
    cinn.cuda()

    dataset = data.MapToSatDataset(args)

    checkpoint_interval  = eval(args['checkpoints']['checkpoint_interval'])
    checkpoint_overwrite = eval(args['checkpoints']['checkpoint_overwrite'])
    checkpoint_on_error  = eval(args['checkpoints']['checkpoint_on_error'])
    figures_interval     = eval(args['checkpoints']['figures_interval'])
    figures_overwrite    = eval(args['checkpoints']['figures_overwrite'])

    N_epochs = eval(args['training']['N_epochs'])
    output_dir = args['checkpoints']['output_dir']

    logfile = open(os.path.join(output_dir, 'losses.dat'), 'w')

    def log_write(string):
        logfile.write(string + '\n')
        print(string, flush=True)


    log_header = '{:>8s}{:>10s}{:>12s}{:>12s}'.format('Epoch', 'Time (m)', 'NLL train', 'NLL val')
    log_fmt    = '{:>8d}{:>10.1f}{:>12.5f}{:>12.5f}'

    log_write(log_header)

    val_x = dataset.val_x.cuda()
    val_y = dataset.val_y.cuda()

    t_start = time.time()

    for epoch in range(N_epochs):

        loss_per_batch = []

        for x, y in dataset.train_loader:
            x, y = x.cuda(), y.cuda()

            nll = cinn.nll(x, y)
            nll.backward()
            loss_per_batch.append(nll.item())

            cinn.optimizer.step()
            cinn.optimizer.zero_grad()

        # end of epoch:

        cinn.eval()

        time_delta = (time.time() - t_start) / 60.
        train_loss = np.mean(loss_per_batch)
        val_loss = cinn.nll(val_x, val_y)

        log_write(log_fmt.format(epoch + 1, time_delta, train_loss, val_loss))
        cinn.train()

    logfile.close()
    cinn.save(os.path.join(output_dir, 'checkpoint_end.pt'))
