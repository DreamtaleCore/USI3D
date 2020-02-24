"""
Scripts for loading the intrinsic image dataset
"""

import random
import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from skimage import io
from skimage.transform import resize


def make_dataset(root, file_name):
    images = []
    pwd_file = os.path.join(root, file_name)

    assert os.path.isfile(pwd_file), '%s is not a valid file pwd' % pwd_file

    with open(pwd_file, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            line = line.strip()
            images.append(os.path.join(root, line))

    return images


def default_loader(path):
    return io.imread(path)


def rgb_to_irg(rgb):
    """ converts rgb to (mean of channels, red chromaticity, green chromaticity) """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 2] = s / 3.0
    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    return irg


def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    irg[..., 2] = rgb[..., 2] / s

    return irg


class CGIFolder(data.Dataset):
    """
    Class for load CGIntrinsic dataset, here shading is pre-computed via CGI and I = A * S
    """

    def __init__(self, params, loader=default_loader, dict_image=None, is_train=True):
        if dict_image is None:
            dict_image = {'input': None, 'albedo': None, 'shading': None, 'mask': None}
        self.dict_image = dict_image
        self.root = params['data_root']
        self.loader = loader
        self.height = params['crop_image_height']
        self.width = params['crop_image_width']
        self.original_size = params['new_size']
        self.rotation_range = 5.0
        self.sigma_chro = 0.025
        self.sigma_I = 0.1
        self.is_train = is_train
        self.half_window = 1
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)
        self.num_scale = 4

        img_dict = {}
        for key in dict_image.keys():
            img_dict[key] = make_dataset(self.root, dict_image[key])

        max_dict = {}
        if os.path.exists(os.path.join(self.root, 'shading', 'shading-max.txt')):
            with open(os.path.join(self.root, 'shading', 'shading-max.txt')) as fid:
                lines = fid.readlines()
                lines = [x.strip() for x in lines]
                for line in lines:
                    items = line.split('\t')
                    if items[0] == 'name' or len(items) != 2:
                        continue
                    max_dict[items[0]] = float(items[1])

        self.max_dict = max_dict

        self.img_dict = img_dict

    def __len__(self):
        return len(self.img_dict['input'])

    def data_argument(self, img, mode, random_pos, random_flip):

        if random_flip > 0.5:
            img = np.fliplr(img)

        # img = rotate(img,random_angle, order = mode)
        img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
        img = resize(img, (self.height, self.width), order=mode)

        return img

    def construct_sub_matrix(self, C):
        h = C.shape[0]
        w = C.shape[1]

        sub_C = np.zeros((9, h - 2, w - 2, 3))
        ct_idx = 0
        for k in range(0, self.half_window * 2 + 1):
            for l in range(0, self.half_window * 2 + 1):
                sub_C[ct_idx, :, :, :] = C[self.half_window + self.Y[k, l]:h - self.half_window + self.Y[k, l], \
                                         self.half_window + self.X[k, l]: w - self.half_window + self.X[k, l], :]
                ct_idx += 1

        return sub_C

    def load_images(self, index, use_da=True):
        len_in = len(self.img_dict['input'])
        len_ab = len(self.img_dict['albedo'])
        len_sd = len(self.img_dict['shading'])
        len_mk = len(self.img_dict['mask'])
        img_in = np.float32(self.loader(self.img_dict['input'][index % len_in])) / 255.
        img_ab = np.float32(self.loader(self.img_dict['albedo'][index % len_ab])) / 255.
        img_sd = np.float32(self.loader(self.img_dict['shading'][index % len_sd])) / 255.
        img_mk = np.float32(self.loader(self.img_dict['mask'][index % len_mk])) / 255.
        file_name = os.path.basename(self.img_dict['input'][index % len_in])

        img_sd = img_sd * self.max_dict[file_name] if file_name in self.max_dict else img_sd
        if len(img_mk.shape) < 3:
            img_mk = np.stack([img_mk, img_mk, img_mk], axis=2)

        ori_h, ori_w = img_in.shape[:2]

        if use_da:
            random_flip = random.random()
            random_start_y = random.randint(0, 9)
            random_start_x = random.randint(0, 9)

            random_pos = [random_start_y, random_start_y + ori_h - 10, random_start_x,
                          random_start_x + ori_w - 10]

            img_in = self.data_argument(img_in, 1, random_pos, random_flip)
            img_ab = self.data_argument(img_ab, 1, random_pos, random_flip)
            img_sd = self.data_argument(img_sd, 1, random_pos, random_flip)
            img_mk = self.data_argument(img_mk, 1, random_pos, random_flip)

        return img_in, img_ab, img_sd, img_mk, file_name

    def construct_R_weights(self, N_feature):
        center_feature = np.repeat(np.expand_dims(N_feature[4, :, :, :], axis=0), 9, axis=0)
        feature_diff = center_feature - N_feature

        r_w = np.exp(- np.sum(feature_diff[:, :, :, 0:2] ** 2, 3) / (self.sigma_chro ** 2)) \
              * np.exp(- (feature_diff[:, :, :, 2] ** 2) / (self.sigma_I ** 2))

        return r_w

    def __getitem__(self, index):
        targets = {}

        img_in, img_ab, img_sd, img_mk, file_name = self.load_images(index, self.is_train)

        img_in[img_in < 1e-4] = 1e-4
        rgb_img = srgb_to_rgb(img_in)
        chromaticity = rgb_to_chromaticity(img_in)
        targets['chromaticity'] = torch.from_numpy(np.transpose(chromaticity, (2, 0, 1))).contiguous().float()

        img_in = torch.from_numpy(np.transpose(img_in, (2, 0, 1))).contiguous().float()
        targets['mask'] = torch.from_numpy(np.transpose(img_mk, (2, 0, 1))).contiguous().float()
        targets['albedo'] = torch.from_numpy(np.transpose(img_ab, (2, 0, 1))).contiguous().float()
        targets['shading'] = torch.from_numpy(np.transpose(img_sd, (2, 0, 1))).contiguous().float()
        targets['name'] = file_name

        for i in range(0, self.num_scale):
            feature_3d = rgb_to_irg(rgb_img)
            sub_matrix = self.construct_sub_matrix(feature_3d)
            r_w = self.construct_R_weights(sub_matrix)
            targets['r_w_s' + str(i)] = torch.from_numpy(r_w).float()
            rgb_img = rgb_img[::2, ::2, :]

        return img_in, targets


class MPIFolder(data.Dataset):
    """
    Class for loading MPI Sentel dataset, here shading is pre-computed via CGI and I = A * S
    """

    def __init__(self, params, transform=None, loader=None, dict_image=None, is_train=True):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, dict_image={},
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
