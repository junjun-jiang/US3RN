# -*- coding: UTF-8 -*-
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
import scipy.io as sio
import random
import torch
import torch.nn.functional as F



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".mat"])


def load_img(filepath):
    x = sio.loadmat(filepath)
    x = x['msi']
    x = torch.tensor(x).float()
    return x

def load_img1(filepath):
    x = sio.loadmat(filepath)
    x = x['RGB']
    x = torch.tensor(x).float()
    return x



class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir1, image_dir2, upscale_factor, patch_size,input_transform=None):
        super(DatasetFromFolder, self).__init__()

        self.patch_size = patch_size
        self.image_filenames1 = [join(image_dir1, x) for x in listdir(image_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.lens = 20000

        self.xs = []
        for img in self.image_filenames1:
            self.xs.append(load_img(img))

        self.ys = []
        for img in self.image_filenames2:
            self.ys.append(load_img1(img))

        self.upscale_factor = upscale_factor
        self.input_transform = input_transform

    def __getitem__(self, index):
        ind = index % 22
        img = self.xs[ind]
        img2 = self.ys[ind]
        upscale_factor = self.upscale_factor
        w = np.random.randint(0, img.shape[0]-self.patch_size)
        h = np.random.randint(0, img.shape[1]-self.patch_size)
        X = img[w:w+self.patch_size, h:h+self.patch_size, :]
        X_1 = img2[w:w+self.patch_size, h:h+self.patch_size, :]
        X_2 = F.interpolate(X.permute(2,0,1).unsqueeze(0), scale_factor=1.0/upscale_factor, mode='bicubic', align_corners=False, recompute_scale_factor=False).squeeze(0).permute(1,2,0)
        Y = F.interpolate(X_1.permute(2,0,1).unsqueeze(0), scale_factor=1.0/upscale_factor, mode='bicubic', align_corners=False, recompute_scale_factor=False).squeeze(0).permute(1,2,0)

        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        # Random rotation
        X = torch.rot90(X, rotTimes, [0,1])
        X_1 = torch.rot90(X_1, rotTimes, [0,1])
        X_2 = torch.rot90(X_2, rotTimes, [0,1])
        Y = torch.rot90(Y, rotTimes, [0,1])

        # Random vertical Flip
        for j in range(vFlip):
            X = X.flip(1)
            X_1 = X_1.flip(1)
            X_2 = X_2.flip(1)
            Y = Y.flip(1)

        # Random Horizontal Flip
        for j in range(hFlip):
            X = X.flip(0)
            X_1 = X_1.flip(0)
            X_2 = X_2.flip(0)
            Y = Y.flip(0)

        X = X.permute(2,0,1)
        X_1 = X_1.permute(2, 0, 1)
        X_2 = X_2.permute(2, 0, 1)
        Y = Y.permute(2, 0, 1)

        return Y, X_1, X_2, X

    def __len__(self):
        return self.lens


class DatasetFromFolder2(data.Dataset):
    def __init__(self, image_dir1, image_dir2, upscale_factor, input_transform=None):
        super(DatasetFromFolder2, self).__init__()
        self.image_filenames1 = [join(image_dir1, x) for x in listdir(image_dir1) if is_image_file(x)]
        self.image_filenames2 = [join(image_dir2, x) for x in listdir(image_dir2) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        self.input_transform = input_transform

        self.xs = []
        self.xs_name = []
        for img in self.image_filenames1:
            self.xs.append(load_img(img))
            self.xs_name.append(img)


        self.ys = []
        for img in self.image_filenames2:
            self.ys.append(load_img1(img))

    def __getitem__(self, index):
        X = self.xs[index]
        Y = self.ys[index]

        upscale_factor = self.upscale_factor

        X_1 = Y
        Y = F.interpolate(X_1.permute(2,0,1).unsqueeze(0), scale_factor=1.0/upscale_factor, mode='bicubic', align_corners=False, recompute_scale_factor=False).squeeze(0).permute(1,2,0)

        X = X.permute(2, 0, 1)
        Y = Y.permute(2, 0, 1)

        return Y, X, self.xs_name[index]

    def __len__(self):
        return len(self.image_filenames1)