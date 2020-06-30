import sys
import glob
from os.path import join

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as T

class UnNormalize:
    def __init__(self, mus, sigmas):
        self.m = mus
        self.s = sigmas
    def __call__(self, x):
        for i in range(x.shape[1]):
            x[:, i] = x[:, i] * self.s[i] + self.m[i]
        return x

class TensorNoTranspose:
    def __call__(self, x):
        return torch.from_numpy(x)

class RandomNpFlip:
    def __call__(self, x):
        out = x
        if random.random() > 0.5:
            out = np.flip(out, 1)
        if random.random() > 0.5:
            out = np.flip(out, 2)
        return np.ascontiguousarray(out)

class NpDequantize:
    def __call__(self, x):
        return x + np.random.random(x.shape).astype(np.float32) / 256.

class RandomNpRotate90:
    def __call__(self, x):
        if random.random() > 0.5:
            x = x.transpose((0,2,1))
            x = np.ascontiguousarray(x)
        return x

class RandomZoomCrop:
    def __init__(self, output_size, window_min=None, window_max=600, deterministic=False):
        self.out_size = output_size
        if not window_min:
            self.w_min = output_size
        else:
            self.w_min = window_min
        self.w_max = window_max
        self.deterministic = deterministic
    def __call__(self, x):
        if self.deterministic:
            w = self.w_max
            n0, m0 = 0,0
        else:
            w = random.randint(self.w_min, self.w_max)
            n0, m0 = (random.randint(0, self.w_max - w) for i in [0,1])
        factor = float(self.out_size) / w
        return zoom(x[:, n0:n0+w, m0:m0+w], (1., factor, factor))

class MatchingFolderDataset(Dataset):
    def __init__(self, matched_files, transform=None):

        self.files = matched_files
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        idx = idx%len(self.files)
        im_A, im_B = (Image.open(f) for f in self.files[idx])

        im_cat = torch.cat([self.to_tensor(im_A), self.to_tensor(im_B)], dim=0).numpy()
        im_transf = self.transform(im_cat)

        return im_transf[:3], im_transf[3:]


class MapToSatDataset:
    def __init__(self, args):

        mu_img    = eval(args['data']['mu_sat'])
        std_img   = eval(args['data']['std_sat'])

        mu_map    = eval(args['data']['mu_map'])
        std_map   = eval(args['data']['std_map'])

        img_dim   = eval(args['data']['rescale_to'])
        crop_min  = eval(args['data']['crop_min'])
        crop_max  = eval(args['data']['crop_max'])

        batchsize = eval(args['training']['batch_size'])
        base_folder = args['data']['data_root_folder']

        self.unnormalize_im  = UnNormalize(mu_img, std_img)
        self.unnormalize_map = UnNormalize(mu_map, std_map)

        self.transf = T.Compose([RandomZoomCrop(img_dim, window_min=crop_min, window_max=crop_max),
                                RandomNpFlip(),
                                RandomNpRotate90(),
                                NpDequantize(),
                                TensorNoTranspose(),
                                T.Normalize(mu_img + mu_map, std_img + std_map)
                               ])

        self.transf_test = T.Compose([RandomZoomCrop(img_dim, window_max=crop_max, deterministic=True),
                                TensorNoTranspose(),
                                T.Normalize(mu_img + mu_map, std_img + std_map)
                               ])

        self.train_files = list(zip(*[sorted(glob.glob(join(base_folder, 'train{}/*'.format(s)))) for s in ['A', 'B']]))
        self.test_files = list(zip(*[sorted(glob.glob(join(base_folder, 'test{}/*'.format(s)))) for s in ['A', 'B']]))

        self.train_files += self.test_files[256 + 32:]
        self.val_files = self.test_files[256:(256 + 32)]
        self.test_files = self.test_files[:256]

        self.train_data = MatchingFolderDataset(self.train_files, self.transf)
        self.test_data  = MatchingFolderDataset(self.test_files, self.transf_test)
        self.val_data =   MatchingFolderDataset(self.val_files, self.transf_test)

        self.val_x = torch.stack([self.val_data[i][0] for i in range(len(self.val_data))])
        self.val_y = torch.stack([self.val_data[i][1] for i in range(len(self.val_data))])

        self.train_loader  = DataLoader(self.train_data, batch_size=batchsize, shuffle=True,
                                   num_workers=8, pin_memory=True, drop_last=True)
        self.test_loader   = DataLoader(self.test_data,  batch_size=batchsize, shuffle=False,
                                   num_workers=4, pin_memory=False, drop_last=False)
