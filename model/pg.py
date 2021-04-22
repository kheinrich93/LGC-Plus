import numpy as np
# import tensorflow as tf
import utils.image_io as image_io
import imageio
import os
from data_generators import TrainingSample
import cv2

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pathlib

net_dir = os.path.split(os.getcwd())[0]

# ConfNet_1600_noFBNexp1_K15
# ConfNet_1600_LF3_K15
# pic = "000004_10"
data_set = 'KITTI-15'
# data_set = 'KITTI-15' Middlebury-v3

sm_method = 'MC_CNN'
# sm_method = 'MC_CNN' ,census

compute_qual = 1
module = 'LGC'
networkname = 'Confmap_LGC_LF_MC_CNN_K15'

compute_WD = 0
# pic = 'Motorcycle'
pic ='000062_10'

data_path = os.path.join(net_dir, 'data', data_set)

left_image_path = os.path.join(data_path, 'images', 'left')
right_image_path = os.path.join(data_path, 'images', 'right')
disp_path = os.path.join(data_path, 'est_'+sm_method)
gt_path = os.path.join(data_path, 'disp_gt')
cv_path = os.path.join(data_path, 'cv_census')

eval_path = os.path.join(os.path.split(net_dir)[0], 'evaluation', 'ConfCorr', networkname)

conf_network = networkname
conf_dir = os.path.join(net_dir, 'results', 'Testing', conf_network)
name_list = os.listdir(conf_dir)

if compute_qual:
    pathlib.Path(eval_path).mkdir(parents=True, exist_ok=True)
    for pic in name_list:
        print('Current Picture: ' + pic)
        pic ='000043_10'
        path_left = os.path.join(left_image_path, pic)
        path_right = os.path.join(right_image_path, pic)
        path_disp = os.path.join(disp_path, pic)
        path_gt = os.path.join(gt_path, pic)

        path_conf = os.path.join(conf_dir, pic, 'ConfMap_' + module + '_' + pic)

        left = image_io.read(path_left + '.png')
        right = image_io.read(path_right + '.png')
        disp = image_io.read(path_disp + '.png')

        if data_set == 'Middlebury-v3':
            gt = image_io.read(path_gt + '.pfm')
        else:
            gt = image_io.read(path_gt + '.png')

        conf = image_io.read(path_conf + '.png')

        res = image_io.get_confidence_correctness_as_image(disp, gt, conf, conf_threshold=0.5, disp_threshold=3)
        jpg_path = os.path.join(eval_path, pic + '_corr.jpg')
        cv2.imwrite(jpg_path, res)

    print('Successfully computed ' + str(len(name_list)) + ' images.')

if compute_WD:
    path_left = os.path.join(left_image_path, pic)
    path_right = os.path.join(right_image_path, pic)
    path_disp = os.path.join(disp_path, pic)

    left = image_io.read(path_left + '.png')
    right = image_io.read(path_right + '.png')
    disp = image_io.read(path_disp + '.png')

    left_warped = image_io.to_warped_image(right, disp, 'r2l')
    warped_errormap = image_io.get_warp_errormap(left, left_warped)
    eval_WD_path = os.path.join(os.path.split(net_dir)[0], 'evaluation', 'WD')
    pathlib.Path(eval_WD_path).mkdir(parents=True, exist_ok=True)
    WD_path = os.path.join(eval_WD_path, pic + '_WD.jpg')
    cv2.imwrite(WD_path, warped_errormap)
    print('Finished errormap:', pic)

# img_png=image_io.read(local_dir2)/255.0
# img_pfm=image_io.read(local_dir)
# test = np.sum(abs(np.subtract(img_pfm,img_png)))
# print("tes")
# image_height=384
# image_width=1280
# test=tf.image.resize_with_crop_or_pad(left_image, image_height, image_width)
# test2=test.numpy()
# img = np.around(np.random.rand(img_size, img_size) * 10)


# from matplotlib import pyplot as plt
# plt.imshow(test2, interpolation='nearest')
# plt.show()
# print(test.shape)
#
# test=loc_image[222][222]
#
# patch_size = 9
# excerpt = np.zeros((patch_size, patch_size))
#
# row = 9
# col = 9
#
# # print(img[0,0])
#
# nb_offset = ((patch_size - 1) / 2)
#
# # Read ground truth and normalise values if necessary
# disp_gt = gt_image
# dimensions = disp_gt.shape
#
# # Check for available ground truth points
# training_samples = []
# for row in range(dimensions[0]):
#     for col in range(dimensions[1]):
#         gt_value = disp_gt[row][col]
#         if (gt_value != 0):
#             # Ground truth point is available -> Create sample for this pixel
#             training_samples.append(TrainingSample(pic, row, col, gt_value))
#
#
# def get_patch(img, row, col, patch_size=9):
#     excerpt = np.zeros((patch_size, patch_size))
#
#     nb_offset = ((patch_size - 1) / 2)
#     for excerpt_row in range(0, patch_size):
#         for excerpt_col in range(0, patch_size):
#             image_row = int(row + excerpt_row - nb_offset)
#             image_col = int(col + excerpt_col - nb_offset)
#             if 0 <= image_row <= (img.shape[1] - 1) and 0 <= image_col <= (img.shape[0] - 1):
#                 # print(img[image_row, image_col],image_row, image_col)
#
#                 excerpt[excerpt_row, excerpt_col] = img[image_row, image_col]
#
#     return excerpt
#
#
# X_disp = np.empty((64, patch_size, patch_size, 1))
# X_local = np.empty((64, patch_size, patch_size, 1))
#
# print(training_samples)
#
# for i, training_sample in enumerate(training_samples):
#     row = training_sample.row
#     col = training_sample.col
#
#     # print(sum(disp_gt))
#     disp_extract = get_patch(disp_image, row, col)
#     local_extract = get_patch(loc_image, row, col)
#
#     #X_disp[i, :, :, 0] = disp_extract
#     X_local[i, :, :, 0] = local_extract
#
# # print(X_disp)
#
# dum = np.ones([patch_size, patch_size])
# print(dum.shape)
#
# X_disp[0, :, :, 0] = dum
#
# # excerpt = np.zeros((patch_size, patch_size))
# # nb_offset = ((patch_size - 1) / 2)
# # img=disp_image
# # for excerpt_row in range(0, patch_size):
# #     for excerpt_col in range(0, patch_size):
# #         image_row = int(215 + excerpt_row - nb_offset)
# #         image_col = int(2 + excerpt_col - nb_offset)
# #         if 0 <= image_row <= (img.shape[1] - 1) and 0 <= image_col <= (img.shape[0] - 1):
# #             excerpt[excerpt_row, excerpt_col] = img[image_row, image_col]

# X_image = np.empty((1, 384, 1280, 3))

# x=tf.zeros((1, 384, 1280, 3), dtype=tf.dtypes.float32, name=None)
# image_dims=x.get_shape().as_list()
# width=image_dims[1]

# s=x.shape()

# print("end")
