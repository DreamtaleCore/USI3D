import argparse
import os
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='/home/ros/ws/pami19/dataset/test/CEILNet/real',
                    help='Path to the dir which need to rename.')
parser.add_argument('--map_name', type=str, default='original_name.txt',
                    help='save the convert log to this file.')
opts = parser.parse_args()


data_dir = opts.dir

sub_names = os.listdir(data_dir)
# sub_names = [x for x in sub_names if os.listdir(os.path.join(data_dir, x))]

try:
    sub_names = sorted(sub_names, key=lambda x: float(x))
except Exception as e:
    sub_names = sorted(sub_names)


with open(os.path.join(data_dir, opts.map_name), 'w') as fid:
    fid.write('class id\toriginal name\n')
    for idx, sub_name in enumerate(sub_names):
        src_pwd = os.path.join(data_dir, sub_name)
        tmps = sub_name.split('.')
        post_fix = '' if len(tmps) <= 1 else '.{}'.format(tmps[-1])
        dst_pwd = os.path.join(data_dir, str(idx) + post_fix)
        shutil.move(src_pwd, dst_pwd)
        line = '{}{}\t{}\n'.format(idx, post_fix, sub_name)
        fid.write(line)

print('Done.')

