import os
import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


DATASET_ROOT = '/media/ros/Files/ws/Dataset/CGIntrinsics-modified'
TRAIN_RATE = 8. / 9.
IS_SUP = False


dir_in = os.path.join(DATASET_ROOT, 'input')
dir_ab = os.path.join(DATASET_ROOT, 'reflectance')
dir_sd = os.path.join(DATASET_ROOT, 'shading')
dir_mk = os.path.join(DATASET_ROOT, 'mask')

image_names = os.listdir(dir_in)
image_names = [x for x in image_names if is_image_file(x)]
image_names.sort()
n_train = int(TRAIN_RATE * len(image_names))

t_bar = tqdm.tqdm(enumerate(image_names))
t_bar.set_description('Processing ')

post_fix = '_sup' if IS_SUP else ''

fid_in_train = open(os.path.join(DATASET_ROOT, 'train-input{}.txt'.format(post_fix)), 'w')
fid_ab_train = open(os.path.join(DATASET_ROOT, 'train-reflectance{}.txt'.format(post_fix)), 'w')
fid_sd_train = open(os.path.join(DATASET_ROOT, 'train-shading{}.txt'.format(post_fix)), 'w')
fid_mk_train = open(os.path.join(DATASET_ROOT, 'train-mask{}.txt'.format(post_fix)), 'w')
fid_in_test = open(os.path.join(DATASET_ROOT, 'test-input.txt'), 'w')
fid_ab_test = open(os.path.join(DATASET_ROOT, 'test-reflectance.txt'), 'w')
fid_sd_test = open(os.path.join(DATASET_ROOT, 'test-shading.txt'), 'w')
fid_mk_test = open(os.path.join(DATASET_ROOT, 'test-mask.txt'), 'w')


for idx, img_name in t_bar:
    if idx < n_train:
        t_bar.set_description('For Train')
        if not IS_SUP:
            if idx % 2 == 0:
                fid_in_train.write(os.path.join('input', img_name) + '\n')
                fid_mk_train.write(os.path.join('mask', img_name) + '\n')
            else:
                fid_ab_train.write(os.path.join('reflectance', img_name) + '\n')
                fid_sd_train.write(os.path.join('shading', img_name) + '\n')
        else:
            if idx % 2 == 0:
                fid_in_train.write(os.path.join('input', img_name) + '\n')
                fid_ab_train.write(os.path.join('reflectance', img_name) + '\n')
                fid_sd_train.write(os.path.join('shading', img_name) + '\n')
                fid_mk_train.write(os.path.join('mask', img_name) + '\n')
    else:
        if IS_SUP:
            continue
        t_bar.set_description('For Test')
        fid_in_test.write(os.path.join('input', img_name) + '\n')
        fid_ab_test.write(os.path.join('reflectance', img_name) + '\n')
        fid_sd_test.write(os.path.join('shading', img_name) + '\n')
        fid_mk_test.write(os.path.join('mask', img_name) + '\n')

print('\nDone.')


