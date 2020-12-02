"""
@author: DremTale
"""
from __future__ import print_function

from tqdm import tqdm

from utils import get_config
from trainer import UnsupIntrinsicTrainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='configs/intrinsic_MIT.yaml', help="net configuration")
parser.add_argument('-i', '--input_dir', type=str, default='/home/ros/datasets/intrinsic/MIT-inner-split/trainA',
                    help="input image path")
parser.add_argument('-o', '--output_folder', type=str, default='id_mit_train-inner-opt',
                    help="output image path")
parser.add_argument('-p', '--checkpoint', type=str, default='checkpoints/mit_inner-opt/gen_00440000.pt',
                    help="checkpoint of MUID")
parser.add_argument('--seed', type=int, default=10, help="random seed")
opts = parser.parse_args()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


wo_fea = 'wo_fea' in opts.checkpoint


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
trainer = UnsupIntrinsicTrainer(config)

state_dict = torch.load(opts.checkpoint, map_location='cuda:0')
trainer.gen_i.load_state_dict(state_dict['i'])
trainer.gen_r.load_state_dict(state_dict['r'])
trainer.gen_s.load_state_dict(state_dict['s'])
trainer.fea_s.load_state_dict(state_dict['fs'])
trainer.fea_m.load_state_dict(state_dict['fm'])

trainer.cuda()
trainer.eval()

if 'new_size' in config:
    new_size = config['new_size']
else:
    new_size = config['new_size_i']

intrinsic_image_decompose = trainer.inference

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
#                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # Make sure the vaule range of input tensor be consistent to the training time
                                   ])

    image_paths = os.listdir(opts.input_dir)
    image_paths = [x for x in image_paths if is_image_file(x)]
    t_bar = tqdm(image_paths)
    t_bar.set_description('Processing')
    for image_name in t_bar:
        image_pwd = os.path.join(opts.input_dir, image_name)

        out_root = os.path.join(opts.output_folder, image_name.split('.')[0])

        if not os.path.exists(out_root):
            os.makedirs(out_root)

        image = Variable(transform(Image.open(image_pwd).convert('RGB')).unsqueeze(0).cuda())

        # Start testing
        im_reflect, im_shading = intrinsic_image_decompose(image, wo_fea)
        im_reflect = (im_reflect + 1) / 2.
        im_shading = (im_shading + 1) / 2.

        path_reflect = os.path.join(out_root, 'output_r.jpg')
        path_shading = os.path.join(out_root, 'output_s.jpg')

        vutils.save_image(im_reflect.data, path_reflect, padding=0, normalize=True)
        vutils.save_image(im_shading.data, path_shading, padding=0, normalize=True)

        if not opts.output_only:
            # also save input images
            vutils.save_image(image.data, os.path.join(out_root, 'input.jpg'), padding=0, normalize=True)

