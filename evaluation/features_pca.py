from os.path import join
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def features_pca(model, dataset, args, path):

    model.eval()
    N_images = 250
    j = 0
    os.makedirs(path, exist_ok=True)

    for x, y in dataset.test_loader:
        y = y.cuda()
        y_np = dataset.map_to_np(y)

        with torch.no_grad():
            feature_pyramid = model.cond_net(y)
            entry_features = model.nice_entry(y)
            feature_pyramid[0] = torch.cat([feature_pyramid[0], entry_features], dim=1)

        for k in range(y.shape[0]):
            plt.imsave(join(path, '{:03d}_y.png'.format(j)), y_np[k])

            for i in range(len(feature_pyramid)):
                ci = feature_pyramid[i][k].cpu().numpy()
                n_features = ci.shape[0]
                size = ci.shape[1]
                margin = max(size//8, 1)

                ci = ci.transpose((1,2,0))
                ci_fit = ci[margin:-margin, margin:-margin, :].reshape(-1, n_features)
                ci = ci.reshape((-1, n_features))

                mu = np.mean(ci_fit, axis=0, keepdims=True)
                sigma = 1. #np.maximum(np.std(ci_fit, axis=0, keepdims=True), 1e-3)

                ci_fit = (ci_fit - mu) / sigma
                ci = (ci - mu) / sigma

                pca = PCA()
                pca.fit(ci_fit)

                ci = pca.transform(ci)
                ci = ci.reshape((size, size, -1))

                gamma = 1.5 + 0.00 * (i-1)**2
                max_quantile = 0.99

                for ch in range(min(min(8 * 2**i, 24), ci.shape[2])):

                    rgb = ci[:, :, ch]
                    rgb = np.sign(rgb) * np.abs(rgb)**gamma

                    rmax = np.quantile(np.abs(rgb[margin:-margin, margin:-margin].flatten()), max_quantile)
                    rmax = max(rmax, 1e-4)

                    plt.imsave(join(path, '{:03d}_{:02d}_{:03d}.png'.format(j, i, ch)), rgb,
                               vmin=-rmax, vmax=rmax, cmap='PRGn')


            j += 1
            print('{}/{}'.format(j, N_images))
            if j >= N_images:
                return
