import os
from os.path import join

import torch
import numpy as np
import matplotlib.pyplot as plt

def sample(model, dataset, args,
           temperature       = None,
           N_examples        = 6,
           N_samples_per_y   = 3,
           test_data         = False,
           test_time_bn      = False,
           big_size          = False,
           save_separate_ims = None):

    base_figsize = 2.5
    temp = temperature

    if temp is None:
        temp = eval(args['testing']['temp'])

    if big_size:
        sample_resolution = dataset.max_size
    else:
        sample_resolution = eval(args['data']['crop_to'])

    resolution_stages = len(eval(args['model']['inn_coupling_blocks']))
    latent_resolution = sample_resolution // (2 ** (resolution_stages - 1))
    latent_channels   = 3 * (4 ** (resolution_stages - 1))

    fig_W = N_samples_per_y + 2
    fig_H = N_examples

    if save_separate_ims is not None:
        os.makedirs(save_separate_ims, exist_ok=True)
        def subplot_imshow(im, counter, im_file):
            plt.imsave(join(save_separate_ims, im_file), im)
            return counter + 1

    else:
        plt.figure(figsize=(base_figsize * fig_W,
                            base_figsize * fig_H))

        def subplot_imshow(im, counter, im_file=None):
            plt.subplot(fig_H, fig_W, counter)
            plt.imshow(im)
            plt.xticks([])
            plt.yticks([])
            return counter + 1

    if big_size:
        data_loader = dataset.hd_loader
    elif test_data:
        data_loader = dataset.test_loader
    else:
        data_loader = [(dataset.val_x, dataset.val_y)]

    model.eval()
    fig_counter = 1

    for x, y in data_loader:

        x, y = x.cuda(), y.cuda()

        x_np = dataset.sat_to_np(x)
        y_np = dataset.map_to_np(y)

        for k in range(x.shape[0]):

            k_example = fig_counter // fig_W
            #if k_example < 32:
                #fig_counter += N_samples_per_y + 2
                #continue
            #print(k_example)

            with torch.no_grad():
                yk = torch.stack([y[k]] * N_samples_per_y, dim=0)
                zk = torch.cuda.FloatTensor(N_samples_per_y, latent_channels,
                                            latent_resolution, latent_resolution)
                zk.normal_(mean=0., std=temp)

                x_gen = model.generate(zk, yk)


            x_gen_np = dataset.sat_to_np(x_gen)
            fig_counter = subplot_imshow(y_np[k], fig_counter, '{:03d}_y.png'.format(k_example))

            for j in range(N_samples_per_y):
                fig_counter = subplot_imshow(x_gen_np[j], fig_counter, '{:03d}_{:03d}.png'.format(k_example, j))

            fig_counter = subplot_imshow(x_np[k], fig_counter, '{:03d}_x.png'.format(k_example))

            if fig_counter > fig_W * fig_H:
                plt.tight_layout()
                model.train()
                return

    model.train()

