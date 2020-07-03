import torch
import numpy as np
import matplotlib.pyplot as plt


def sample(model, dataset, args, test_data=False):
    N_examples          = 6
    N_samples_per_image = 3
    base_figsize        = 2.5
    #sample_resolution  = 512
    sample_resolution   = eval(args['data']['crop_to'])
    temp                = eval(args['testing']['temp'])

    resolution_stages = len(eval(args['model']['inn_coupling_blocks']))
    latent_resolution = sample_resolution // (2 ** (resolution_stages - 1))
    latent_channels = 3 * (4 ** (resolution_stages - 1))

    fig_W = N_samples_per_image + 2
    fig_H = N_examples

    fig_counter = 1

    plt.figure(figsize=(base_figsize * fig_W,
                        base_figsize * fig_H))

    def subplot_imshow(im, counter):
        plt.subplot(fig_H, fig_W, counter)
        plt.imshow(im)
        plt.xticks([])
        plt.yticks([])
        return counter + 1

    if test_data:
        data_loader = dataset.test_loader
    else:
        data_loader = [(dataset.val_x, dataset.val_y)]

    model.eval()

    for x, y in data_loader:

        x, y = x.cuda(), y.cuda()

        x_np = dataset.sat_to_np(x)
        y_np = dataset.map_to_np(y)

        for k in range(x.shape[0]):

            with torch.no_grad():
                yk = torch.stack([y[k]] * N_samples_per_image, dim=0)
                zk = torch.cuda.FloatTensor(N_samples_per_image, latent_channels,
                                            latent_resolution, latent_resolution)
                zk.normal_(mean=0., std=temp)

                x_gen = model.generate(zk, yk)

            x_gen_np = dataset.sat_to_np(x_gen)
            fig_counter = subplot_imshow(y_np[k], fig_counter)

            for j in range(N_samples_per_image):
                fig_counter = subplot_imshow(x_gen_np[j], fig_counter)

            fig_counter = subplot_imshow(x_np[k], fig_counter)

            if fig_counter > fig_W * fig_H:
                plt.tight_layout()
                model.train()
                return

    model.train()

