#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import sys

from test import Test
from data_generators import DataSample

import utils.tf_specs as tf_specs

parser = argparse.ArgumentParser(description='Argument parser')

"""Arguments related to input, monitoring and outputs"""
parser.add_argument('--module_name', dest='module_name', type=str, default='CVA', help='CVA, ConfNet, LGC')
parser.add_argument('--model_dir', dest='model_dir', type=str, default='CVA-Net_test', help='location of model')
parser.add_argument('--data_dir', dest='data_dir', type=str, default='data', help='data')
parser.add_argument('--data_set', dest='data_set', type=str, default='KITTI-12', help='data_set')
parser.add_argument('--result_dir', dest='result_dir', type=str, default='results', help='results')
parser.add_argument('--experiment', dest='experiment', type=str, default='', help='location of experiment')
parser.add_argument('--sm_method', dest='sm_method', type=str, default='census', help='stereo matching method: census or MC_CNN')

"""Arguments related to export"""
parser.add_argument('--save_disp_map', dest='save_disp_map', type=bool, default=False, help='save disp map')
parser.add_argument('--save_png', dest='save_png', type=bool, default=True, help='save as .png')

"""Arguments related to testing"""
parser.add_argument('--image_amount', dest='image_amount', type=int, default=1, help='number of images to be tested')
parser.add_argument('--start_at', dest='start_at', type=int, default=0, help='start testing at image number')
parser.add_argument('--end_at', dest='end_at', type=int, default=0, help='end testing at image number')
parser.add_argument('--check_epoch', dest='check_epoch', type=str, default='', help='check weight at epoch')

"""Arguments related to respective module"""
parser.add_argument('--LGC_local_input', dest='LGC_local_input', type=str, default='Confmap_ConfNet_1600e',
                    help='name of local input in testing folder')
parser.add_argument('--LGC_global_input', dest='LGC_global_input', type=str, default='Confmap_ConfNet_1600e',
                    help='name of global input in testing folder')
parser.add_argument('--ConfNet_resize', dest='ConfNet_resize', type=str, default='interpol', help='interpol or pad')

"""Arguments related to Tensorflow 2.x"""
parser.add_argument('--USE_GPU', dest='USE_GPU', type=bool, default=False, help='use tf-lib for GPU-computation')
parser.add_argument('--GPU_MEM_LIMIT', dest='GPU_MEM_LIMIT', type=int, default=8000, help='tf memory limit')

args = parser.parse_args()

tf_specs.specify_tf(args.USE_GPU, args.GPU_MEM_LIMIT)

net_dir = os.path.split(os.getcwd())[0]
module_name = args.module_name
model_dir = args.model_dir

# pathing
data_path = os.path.join(net_dir, args.data_dir, args.data_set)

left_image_path = os.path.join(data_path, 'images', 'left')
right_image_path = os.path.join(data_path, 'images', 'right')
disp_path = os.path.join(data_path, 'est_'+args.sm_method)
gt_path = os.path.join(data_path, 'disp_gt')
cv_path = os.path.join(data_path, 'cv_'+args.sm_method)
local_dir = os.path.join(net_dir, args.result_dir, 'Testing', args.LGC_local_input)
global_dir = os.path.join(net_dir, args.result_dir, 'Testing', args.LGC_global_input)
model_path = os.path.join(net_dir, 'results', 'Training', args.experiment, model_dir)
results_path = os.path.join(net_dir, args.result_dir, 'Testing', 'Confmap_' + model_dir)

weight_path = os.path.join(model_path, 'models')

weight_name = ''
weight_list = os.listdir(weight_path)

# Consider latest epoch, if no epoch is stated
if not args.check_epoch:
    weight_nr = 0
    for weight in weight_list:
        if int(weight.split('_')[1]) > weight_nr:
            weight_name = weight
            weight_nr = int(weight.split('_')[1])
else:
    for weight in weight_list:
        if weight.split('_')[1] == args.check_epoch:
            weight_name = weight

if not weight_name:
    print("ERROR: weight for epoch ", args.check_epoch, " does not exist.")
    sys.exit(0)

weight_path = os.path.join(weight_path, weight_name)
param_dir = os.path.join(model_path, 'parameter')

if len(os.listdir(cv_path)[0].split('.')) > 1:
    cv_ending = "." + os.listdir(cv_path)[0].split('.')[1]
else:
    cv_ending = ""

loc_path = ""
glob_path = ""
cv_depth = ""
cv_sample = ""

samples = []

# Set range of image(s) to test, according to arguments
name_list = os.listdir(left_image_path)
if args.image_amount != 0:
    end_at = args.image_amount
else:
    if args.end_at == 0:
        end_at = len(os.listdir(left_image_path))
    else:
        end_at = args.end_at

name_list = name_list[args.start_at:end_at]

# Set sample
for img_idx in name_list:
    sample_name = img_idx.split('.')[0]
    if args.module_name == "LGC":
        loc = os.path.join(local_dir, sample_name)
        glob = os.path.join(global_dir, sample_name)
        loc_path = os.path.join(loc, os.listdir(loc)[0])
        glob_path = os.path.join(glob, os.listdir(glob)[0])

    if args.module_name == "CVA":
        cv_depth = 256
        cv_sample = os.path.join(cv_path, sample_name) + cv_ending

    samples.append(DataSample(isTrainingSet=False, gt_path=os.path.join(gt_path, sample_name) + '.png',
                              left_image_path=os.path.join(left_image_path, sample_name) + '.png',
                              right_image_path=os.path.join(right_image_path, sample_name) + '.png',
                              disp_path=os.path.join(disp_path, sample_name) + '.png',
                              local_path=loc_path,
                              global_path=glob_path,
                              cost_volume_depth=cv_depth,
                              cost_volume_path=cv_sample,
                              result_path=os.path.join(results_path, sample_name)))

tester = Test(weights_file=weight_path, param_file=param_dir,
              module_name=module_name, save_disp_map=args.save_disp_map, ConfNet_resize=args.ConfNet_resize)
tester.predict(samples)

print("Finished testing", end_at, "image[s]")
