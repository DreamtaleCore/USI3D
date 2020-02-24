import argparse
import numpy as np
import skimage
from skimage import io
from skimage.morphology import square
import os
import cv2
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dir_src', type=str,
                    default='/media/ros/Files/ws/Dataset/CGIntrinsics/intrinsics_final/images', help="the src dir")
parser.add_argument('--dir_dst', type=str,
                    default='/media/ros/Files/ws/Dataset/CGIntrinsics-modified', help="the dst dir")
opts = parser.parse_args()


def check_dir(s_dir):
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)


def convert_dataset(data_root, img_name):
    """
    todo: See image_folder@CGIntrinsicsImageFolder
    :param data_root:
    :param img_name:
    :return:
    """
    pwd_in = os.path.join(data_root, img_name)
    pwd_ab = os.path.join(data_root, img_name.replace('.png', '_albedo.png'))
    pwd_mk = os.path.join(data_root, img_name.replace('.png', '_mask.png'))

    srgb_img = np.float32(io.imread(pwd_in)) / 255.0
    gt_R = np.float32(io.imread(pwd_ab)) / 255.0
    mask = np.float32(io.imread(pwd_mk)) / 255.0

    gt_R_gray = np.mean(gt_R, 2)
    mask[gt_R_gray < 1e-6] = 0
    mask[np.mean(srgb_img, 2) < 1e-6] = 0

    mask = skimage.morphology.binary_erosion(mask, square(11))
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    gt_R[gt_R < 1e-6] = 1e-6

    rgb_img = srgb_img ** 2.2
    rgb_img = np.clip(rgb_img, 0, 1)
    gt_S = rgb_img / gt_R

    mask[gt_S > 10] = 0
    gt_S[gt_S > 20] = 20
    mask[gt_S < 1e-4] = 0
    gt_S[gt_S < 1e-4] = 1e-4

    if np.sum(mask) < 10:
        max_S = 1.0
    else:
        max_S = np.percentile(gt_S[mask > 0.5], 95)

    gt_S = gt_S / max_S

    gt_S = np.clip(gt_S, 0, 1)

    return rgb_img, gt_R, gt_S, mask, max_S


def proc_dataset(src_dir, dst_dir):
    sub_dirs = os.listdir(src_dir)
    sub_dirs = [x for x in sub_dirs if os.path.isdir(os.path.join(src_dir, x))]
    sub_dirs.sort()
    t_bar = tqdm.tqdm(sub_dirs)

    dst_dir_input = os.path.join(dst_dir, 'input')
    dst_dir_reflection = os.path.join(dst_dir, 'reflectance')
    dst_dir_shading = os.path.join(dst_dir, 'shading')
    dst_dir_mask = os.path.join(dst_dir, 'mask')

    check_dir(dst_dir_input)
    check_dir(dst_dir_reflection)
    check_dir(dst_dir_shading)
    check_dir(dst_dir_mask)

    with open(os.path.join(dst_dir_shading, 'shading-max.txt'), 'w') as fid:
        fid.write('name\tmax\n')
        for sub_name in t_bar:
            sub_dir = os.path.join(src_dir, sub_name)
            t_bar.set_description(sub_name)

            image_names = os.listdir(sub_dir)
            image_names = [x for x in image_names if 'albedo' not in x and 'mask' not in x]

            for image_name in image_names:
                rgb_image, gt_refl, gt_shading, mask, max_S = convert_dataset(sub_dir, image_name)

                new_image_name = '{}-{}'.format(sub_name, image_name)

                cv2.imwrite(os.path.join(dst_dir_input, new_image_name), np.uint8(rgb_image * 255))
                cv2.imwrite(os.path.join(dst_dir_reflection, new_image_name), np.uint8(gt_refl * 255))
                fid.write('{}\t{}\n'.format(new_image_name, max_S))
                cv2.imwrite(os.path.join(dst_dir_shading, new_image_name), np.uint8(gt_shading * 255))
                cv2.imwrite(os.path.join(dst_dir_mask, new_image_name), np.uint8(mask) * 255)


if __name__ == '__main__':
    proc_dataset(opts.dir_src, opts.dir_dst)

