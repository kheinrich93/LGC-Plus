#!/usr/bin/env python
# coding: utf-8

import argparse
import time
import os.path
import sys

import params
from data_generators import DataSample
from train import Train

# import from /utils
import utils.util as util
import utils.tf_specs as tf_specs

parser = argparse.ArgumentParser(description='Argument parser')
"""Arguments related to input, monitoring and outputs"""
parser.add_argument('--module_name', dest='module_name', type=str, default='CVA', help='CVA, ConfNet, LGC, LFN ')
parser.add_argument('--network_name', dest='network_name', type=str, default='MMNet-module', help='name of the network')
parser.add_argument('--data_dir', dest='data_dir', type=str, default='data', help='data')
parser.add_argument('--data_set', dest='data_set', type=str, default='KITTI-12', help='data_set')
parser.add_argument('--sm_method', dest='sm_method', type=str, default='census', help='stereo matching method: census or MC-CNN')
parser.add_argument('--result_dir', dest='result_dir', type=str, default='results', help='results')
parser.add_argument('--output_path', dest='output_path', type=str, default='Training', help='dir of output in /result')
parser.add_argument('--plot_graph', dest='plot_graph', type=bool, default=False, help='plot modelgraph')

"""Arguments related to training"""
parser.add_argument('--epoch', dest='epoch', type=int, default=14, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in patches')
parser.add_argument('--amount_training_data', dest='amount_training_data', type=int, default='1',
                    help='# of images/cv for training')
parser.add_argument('--amount_validation_data', dest='amount_validation_data', type=int, default='1',
                    help='# of images/cv for validation')
parser.add_argument('--experiment', dest='experiment', type=str, default='', help='location of experiment')

"""Arguments related to ConfNet and CVA"""
parser.add_argument('--crop_height', dest='crop_height', type=int, default=256, help='crop height')
parser.add_argument('--crop_width', dest='crop_width', type=int, default=512, help='crop width')
parser.add_argument('--CVA_data_mode', dest='CVA_data_mode', type=str, default='cv', help='image or cv')
parser.add_argument('--use_BN', dest='use_BN', type=str, default=True, help='BN in decoder for ConfNet')

"""Arguments related to LGC input"""
parser.add_argument('--LGC_local_input', dest='LGC_local_input', type=str, default='',
                    help='name of local input in testing folder')
parser.add_argument('--LGC_global_input', dest='LGC_global_input', type=str, default='',
                    help='name of global input in testing folder')
parser.add_argument('--use_warp', dest='use_warp', type=str, default='', help='use warped image: EF or LF')

"""Arguments related to Tensorflow 2.x"""
parser.add_argument('--USE_GPU', dest='USE_GPU', type=bool, default=False, help='use tf-lib for GPU-computation')
parser.add_argument('--GPU_MEM_LIMIT', dest='GPU_MEM_LIMIT', type=int, default=8000, help='tf memory limit')

args = parser.parse_args()

tf_specs.specify_tf(args.USE_GPU, args.GPU_MEM_LIMIT)

parameter = params.Params()

parameter.use_warp = args.use_warp

parameter.epochs = args.epoch
parameter.batch_size = args.batch_size

parameter.CNN_mode = "Training"
parameter.use_BN = args.use_BN

# image: Compute cost volume extracts on the fly (needs more CPU power)
# cv: Load precomputed cost volumes (needs more RAM)
parameter.module_name = args.module_name
parameter.CVA_data_mode = args.CVA_data_mode

if not args.crop_width % 16 == 0 or not args.crop_height % 16 == 0:
    print("ERROR: Crop height or width is not an integer divisor of 16.")
    sys.exit(0)

parameter.crop_height = args.crop_height
parameter.crop_width = args.crop_width
parameter.plot_graph = args.plot_graph

# ---------------------------
# Assemble datasets
# ---------------------------
parameter.training_data = []
parameter.validation_data = []

net_dir = os.path.split(os.getcwd())[0]

# pathing
data_path = os.path.join(net_dir, args.data_dir, args.data_set)

left_image_path = os.path.join(data_path, 'images', 'left')
right_image_path = os.path.join(data_path, 'images', 'right')
disp_path = os.path.join(data_path, 'est_'+args.sm_method)
gt_path = os.path.join(data_path, 'disp_gt')
cv_path = os.path.join(data_path, 'cv_'+args.sm_method)

local_dir = os.path.join(net_dir, args.result_dir, 'Testing', args.LGC_local_input)
global_dir = os.path.join(net_dir, args.result_dir, 'Testing', args.LGC_global_input)

output_dir = os.path.join(net_dir, args.result_dir, args.output_path)

# delete tf images
util.delete_summary_images()
if len(os.listdir(cv_path)[0].split('.')) > 1:
    cv_ending = "." + os.listdir(cv_path)[0].split('.')[1]
else:
    cv_ending = ""

loc_path = ""
glob_path = ""
cv_depth = ""
cv_sample = ""

for img_idx in range(0, args.amount_training_data):
    sample_name = str(img_idx)
    while len(sample_name) < 6:
        sample_name = '0' + sample_name
    sample_name += '_10'

    if args.module_name == "LGC":
        loc = os.path.join(local_dir, sample_name)
        glob = os.path.join(global_dir, sample_name)
        loc_path = os.path.join(loc, os.listdir(loc)[0])
        glob_path = os.path.join(glob, os.listdir(glob)[0])

    if args.module_name == "CVA":
        cv_depth = 256
        cv_sample = os.path.join(cv_path, sample_name) + cv_ending

    parameter.training_data.append(DataSample(isTrainingSet=True, gt_path=os.path.join(gt_path, sample_name) + '.png',
                                              left_image_path=os.path.join(left_image_path, sample_name) + '.png',
                                              right_image_path=os.path.join(right_image_path, sample_name) + '.png',
                                              disp_path=os.path.join(disp_path, sample_name) + '.png',
                                              local_path=loc_path,
                                              global_path=glob_path,
                                              cost_volume_depth=cv_depth,
                                              cost_volume_path=cv_sample
                                              ))

for img_idx in range(args.amount_training_data, (args.amount_training_data + args.amount_validation_data)):
    sample_name = str(img_idx)
    while len(sample_name) < 6:
        sample_name = '0' + sample_name
    sample_name += '_10'

    if args.module_name == "LGC":
        loc = os.path.join(local_dir, sample_name)
        glob = os.path.join(global_dir, sample_name)
        loc_path = os.path.join(loc, os.listdir(loc)[0])
        glob_path = os.path.join(glob, os.listdir(glob)[0])

    if args.module_name == "CVA":
        cv_depth = 256
        cv_sample = os.path.join(cv_path, sample_name) + cv_ending

    parameter.validation_data.append(
        DataSample(isTrainingSet=False, gt_path=os.path.join(gt_path, sample_name) + '.png',
                   left_image_path=os.path.join(left_image_path, sample_name) + '.png',
                   right_image_path=os.path.join(right_image_path, sample_name) + '.png',
                   disp_path=os.path.join(disp_path, sample_name) + '.png',
                   local_path=loc_path,
                   global_path=glob_path,
                   cost_volume_depth=cv_depth,
                   cost_volume_path=cv_sample
                   ))

print('Training and validation-dataset built...')
# ---------------------------
# Start training
# ---------------------------
experiment_series = 'dynamic-depth'

if not args.network_name:
    network_name = args.module_name.format(int(time.time()))
else:
    network_name = args.network_name

pretrained_network = ''

print('Initialising...')
trainer = Train(parameter=parameter, network_name=network_name, experiment_series=args.experiment,
                root_dir=output_dir, pretrained_network=pretrained_network)

print('Start training...')

trainer.train()

print('Finished training of', args.network_name)
