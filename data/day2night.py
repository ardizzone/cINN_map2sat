import sys
import glob
from os.path import join

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as T

TRAIN_SET_OVERSAMPLING = 1

class UnNormalize:
    def __init__(self, mus, sigmas):
        self.m = mus
        self.s = sigmas
    def __call__(self, x_orig):
        x = x_orig.clone()
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

class RandomZoomCropRotate:
    def __init__(self, output_size, scale_min=1., scale_max=1., shear=0.1, rot_max=0.00, deterministic=False):

        self.deterministic = deterministic
        self.w         = output_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rot_max   = rot_max
        self.shear     = shear

    def _random_valid_crop(self, x, w, margin=0):
        if self.deterministic:
            n0, m0 = margin, margin
        else:
            n0, m0 = (random.randint(margin, x.shape[i+1] - w - margin) for i in [0,1])
        return x[:, n0:n0+w, m0:m0+w]

    def __call__(self, x):
        if self.deterministic:
            transf_matrix = 2 * np.eye(2)
            margin = 0
        else:
            a = random.uniform(-self.rot_max, self.rot_max) * np.pi
            rc, rs = np.cos(a), np.sin(a)
            sx = random.choice([1, -1]) * random.uniform(self.scale_min, self.scale_max)
            sy = random.choice([1, -1]) * random.uniform(self.scale_min, self.scale_max)
            hx = random.uniform(-self.shear, self.shear)
            hy = random.uniform(-self.shear, self.shear)

            # geometry <3
            #margin = 2. - 1. / (abs(rc) + abs(rs)) - min(abs(sx), abs(sy))
            #margin = int(self.w * 1.5 * margin)
            #margin = min(self.w // 2, margin)
            margin = 0

            rot_matrix = np.array([[rc, rs],
                                   [-rs, rc]])

            scale_shear = np.array([[1. / sx, hx /sx],
                                    [hy / sy, 1./sy]])

            transf_matrix = scale_shear @ rot_matrix

        x = self._random_valid_crop(x, min(int(2.3 * self.w), x.shape[1]), 0)

        for i in range(x.shape[0]):
            x[i] = affine_transform(x[i], transf_matrix, mode='mirror')

        x = self._random_valid_crop(x, self.w, margin)

        return x

class DualImageDataset(Dataset):
    def __init__(self, files, transform=None):

        self.files = files
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        idx = idx%len(self.files)
        im_both = self.to_tensor(Image.open(self.files[idx]))

        im_A, im_B = im_both[:, :, :256], im_both[:, :, 256:]

        im_cat = torch.cat([im_A, im_B], dim=0).numpy()
        im_transf = self.transform(im_cat)

        return im_transf[:3], im_transf[3:]


class DayToNightDataset:
    def __init__(self, args):
        reverse   = (args['data']['dataset'] == 'NIGHT2DAY')

        mu_img    = eval(args['data']['mu_sat'])
        std_img   = eval(args['data']['std_sat'])

        mu_map    = eval(args['data']['mu_map'])
        std_map   = eval(args['data']['std_map'])

        img_dim   = eval(args['data']['crop_to'])
        scale_min = eval(args['data']['scale_min'])
        scale_max = eval(args['data']['scale_max'])
        shear     = eval(args['data']['shear_max'])

        batchsize = eval(args['training']['batch_size'])
        base_folder = args['data']['data_root_folder']

        self.max_size = 256

        if reverse:
            mu_img, mu_map = mu_map, mu_img
            std_img, std_map = std_map, std_img

        self.unnormalize_im  = UnNormalize(mu_img, std_img)
        self.unnormalize_map = UnNormalize(mu_map, std_map)

        self.transf = T.Compose([
                                RandomZoomCropRotate(img_dim, scale_min, scale_max, shear),
                                NpDequantize(),
                                TensorNoTranspose(),
                                T.Normalize(mu_img + mu_map, std_img + std_map)
                               ])

        self.transf_test = T.Compose([
                                RandomZoomCropRotate(img_dim, deterministic=True),
                                TensorNoTranspose(),
                                T.Normalize(mu_img + mu_map, std_img + std_map)
                               ])

        self.transf_hd = T.Compose([
                                TensorNoTranspose(),
                                T.Normalize(mu_img + mu_map, std_img + std_map)
                               ])


        self.test_files  = list(sorted(glob.glob(join(base_folder, 'test_small/*'))))

        train_images = (list(sorted(glob.glob(join(base_folder, 'val/*'))))
                      + list(sorted(glob.glob(join(base_folder, 'train/*')))))

        random.seed(0)
        random.shuffle(train_images)
        random.seed()


        self.val_files   = train_images[:32]
        self.train_files = train_images[32:]

        self.train_files = self.train_files * TRAIN_SET_OVERSAMPLING

        self.train_data = DualImageDataset(self.train_files, self.transf)
        self.test_data  = DualImageDataset(self.train_files, self.transf_test)
        self.hd_data    = DualImageDataset(self.test_files, self.transf_hd)
        self.val_data   = DualImageDataset(self.val_files, self.transf_test)

        val_imgs = [self.val_data[i] for i in range(len(self.val_data))]
        self.val_x = torch.stack([v[0] for v in val_imgs])
        self.val_y = torch.stack([v[1] for v in val_imgs])

        self.train_loader  = DataLoader(self.train_data, batch_size=batchsize, shuffle=True,
                                   num_workers=14, pin_memory=True, drop_last=True)
        self.test_loader   = DataLoader(self.test_data,  batch_size=batchsize, shuffle=False,
                                   num_workers=4, pin_memory=False, drop_last=False)
        self.hd_loader     = DataLoader(self.hd_data,    batch_size=1, shuffle=False,
                                   num_workers=1, pin_memory=False, drop_last=False)

        self.epoch_length = len(self.train_loader)

    def map_to_np(self, y):
        y = self.unnormalize_map(y)
        y = y.data.cpu().numpy()
        y = y.transpose((0, 2, 3, 1))
        y = np.clip(y, 0., 1.)
        return y

    def sat_to_np(self, x):
        x = self.unnormalize_im(x)
        x = x.data.cpu().numpy()
        x = x.transpose((0, 2, 3, 1))
        x = np.clip(x, 0., 1.)
        return x
