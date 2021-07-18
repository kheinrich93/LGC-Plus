import math
import os

import numpy as np
import tensorflow as tf

# change imports for depending on os (for cluster usage)
from platform import system

if system() == 'Windows':
    import pickle
else:
    import pickle5 as pickle

import graph
import census_metric
import cost_volume

# import from /utils
import utils.image_io as image_io


class Test:

    def __init__(self, weights_file, param_file, module_name, extract_width=100, extract_height=100,
                 file_extension='.pfm', save_png=True, save_disp_map=False, ConfNet_resize='interpol'):
        self.module_name = module_name
        self.save_png = save_png
        self.file_extension = file_extension
        self.save_disp_map = save_disp_map
        self.ConfNet_resize = ConfNet_resize

        # Depending on the amount of available memory, it may be necessary to process the cost volume block-wise
        self.extract_width = extract_width
        self.extract_height = extract_height

        # self.cv_norm = [0.0, 1.0]
        # Load parameter

        with open(param_file, 'rb') as param_file:
            parameter = pickle.load(param_file)
        self.neighbourhood_size = parameter.nb_size
        self.cost_volume_depth = parameter.cost_volume_depth
        self.loss_type = parameter.loss_type
        self.cv_norm = [0.0, 1.0]

        if not hasattr(parameter, 'use_warp'):
            parameter.use_warp = False

        if not hasattr(parameter, 'use_BN'):
            parameter.use_BN = True

        self.use_warp = parameter.use_warp
        parameter.CNN_mode = 'Testing'

        # Load trained model
        if module_name == 'CVA':
            self.model = graph.CVANet().get_model(parameter)
        elif module_name == 'ConfNet':
            self.model = graph.ConfNet().get_model(parameter)
        elif module_name == 'LGC':
            self.model = graph.LGC().get_model(parameter)
        elif module_name == 'LFN':
            self.model = graph.LFN().get_model(parameter)
        self.model.load_weights(weights_file)

    def create_cost_volume(self, sample):
        image_left = image_io.read(sample.left_image_path)
        image_right = image_io.read(sample.right_image_path)

        cm = census_metric.CensusMetric(5, 5)
        census_left = cm.__create_census_trafo__(image_left)
        census_right = cm.__create_census_trafo__(image_right)
        return cm.__compute_cost_volume__(census_left, census_right, self.cost_volume_depth)

    def load_cost_volume(self, sample):
        cv_path = sample.cost_volume_path
        cv = cost_volume.CostVolume()
        if cv_path[-3:] == 'bin':
            img_shape = image_io.read(sample.left_image_path).shape
            cv.load_bin(cv_path, img_shape[0], img_shape[1], self.cost_volume_depth)

        elif cv_path[-3:] == 'dat':
            cv.load_dat(cv_path)

        else:
            with open(cv_path, 'rb') as file:
                cv = pickle.load(file)

        if sample.cost_volume_depth > self.cost_volume_depth:
            cv.reduce_depth(self.cost_volume_depth)

        cv.normalise(self.cv_norm[0], self.cv_norm[1])
        return cv

    def read_tf_image(self, image_path, shape=None, dtype=tf.uint8):
        image_raw = tf.io.read_file(image_path)

        if dtype == tf.uint8:
            image = tf.image.decode_image(image_raw)
        else:
            image = tf.image.decode_png(image_raw, dtype=dtype)

        if shape is None:
            image.set_shape([None, None, 3])
        else:
            image.set_shape(shape)

        # set size to factorised size
        size_factor = 16
        size = image.shape

        # interpolated or crop resize
        if self.ConfNet_resize == 'interpol':
            image_height = math.floor(size[0] / size_factor) * size_factor
            image_width = math.floor(size[1] / size_factor) * size_factor
            image = tf.image.resize(image, [image_height, image_width], method='bilinear')
        elif self.ConfNet_resize == 'pad':
            image_height = math.ceil(size[0] / size_factor) * size_factor
            image_width = math.ceil(size[1] / size_factor) * size_factor
            image = tf.image.resize_with_crop_or_pad(image, image_height, image_width)

        return image

    @staticmethod
    def get_patch(img_list, row, col, patch_size=9):
        img = img_list[0]
        excerpt = np.zeros((patch_size, patch_size))
        nb_offset = ((patch_size - 1) / 2)

        for excerpt_row in range(0, patch_size):
            for excerpt_col in range(0, patch_size):
                image_row = int(row + excerpt_row - nb_offset)
                image_col = int(col + excerpt_col - nb_offset)

                if 0 <= image_row <= (img.shape[0] - 1) and 0 <= image_col <= (img.shape[1] - 1):
                    excerpt[excerpt_row, excerpt_col] = img[image_row, image_col]

        return excerpt

    def predict(self, samples):
        for idx, sample in enumerate(samples):
            print('Started sample ' + str(idx + 1) + ' of ' + str(len(samples)))

            if self.module_name == 'CVA':
                # Compute / load the cost volume (in this example the cost volume is computed based on the Census metric)
                if sample.cost_volume_path:
                    print('    Load cost volume...')
                    cv = self.load_cost_volume(sample)
                else:
                    print('    Compute cost volume...')
                    cv = self.create_cost_volume(sample)

                border = int((self.neighbourhood_size - 1) / 2)
                cv_data = cv.get_data(border)
                cost_volume_dims = cv.dim()

                # Process cost volume block-wise to get the confidence map
                print('    Compute confidence map...')

                confidence_map = np.zeros((cost_volume_dims[0], cost_volume_dims[1]))
                start_y = 0
                end_y = start_y + self.extract_height
                while start_y < cost_volume_dims[0]:
                    start_x = 0
                    end_x = start_x + self.extract_width

                    while (start_x < cost_volume_dims[1]):
                        if (end_x > cost_volume_dims[1]):
                            end_x = cost_volume_dims[1]
                        if (end_y > cost_volume_dims[0]):
                            end_y = cost_volume_dims[0]

                        extract = cv_data[start_y:end_y + 2 * border, start_x:end_x + 2 * border, :]
                        net_input = np.empty((1, extract.shape[0], extract.shape[1], extract.shape[2], 1))
                        net_input[0, :, :, :, 0] = extract

                        confidence_map[start_y:end_y, start_x:end_x] = self.model.predict(net_input)[0, :, :, 0, 0]

                        start_x = start_x + self.extract_width
                        end_x = start_x + self.extract_width

                    start_y = start_y + self.extract_height
                    end_y = start_y + self.extract_height

            elif self.module_name == 'ConfNet':

                # Load images
                img_shape = image_io.read(sample.left_image_path).shape
                left = tf.cast(self.read_tf_image(sample.left_image_path, [None, None, 3]), tf.float32)
                right = tf.cast(self.read_tf_image(sample.right_image_path, [None, None, 3]), tf.float32)
                disp = tf.cast(self.read_tf_image(sample.disp_path, [None, None, 1], dtype=tf.uint16),
                               tf.float32) / 256.0

                disp = tf.expand_dims(disp, axis=0)

                # Reverse to compute warped difference
                rightRGB = tf.reverse(right, axis=[-1])
                leftRGB = tf.reverse(left, axis=[-1])

                # Set input according to fusion module
                if self.use_warp == 'EF':
                    left_warped = image_io.to_warped_image(rightRGB.numpy(), np.squeeze(disp.numpy()), 'r2l')
                    warped_errormap = image_io.get_warp_errormap(leftRGB.numpy(), left_warped)
                    warped_errormap2 = tf.cast(tf.expand_dims(warped_errormap, axis=-1), tf.float32)
                    image = tf.concat((leftRGB, warped_errormap2), axis=-1)
                else:
                    image = left

                if self.use_warp == 'LF':
                    left_warped = image_io.to_warped_image(rightRGB.numpy(), np.squeeze(disp.numpy()), 'r2l')
                    warped_errormap = image_io.get_warp_errormap(leftRGB.numpy(), left_warped)
                    warped_errormap = tf.expand_dims(warped_errormap, axis=0)
                    warped_errormap = tf.cast(tf.expand_dims(warped_errormap, axis=-1), tf.float32)

                image = tf.expand_dims(image, axis=0)

                if self.use_warp == 'LF':
                    confidence_map_pred = self.model.predict(x=(image, disp, warped_errormap))[0, :, :, 0]
                else:
                    confidence_map_pred = self.model.predict(x=(image, disp))[0, :, :, 0]

                confidence_map_pred = tf.expand_dims(confidence_map_pred, axis=-1)

                # interpol or padding
                if self.ConfNet_resize == 'interpol':
                    confidence_map = tf.image.resize(confidence_map_pred, [img_shape[0], img_shape[1]],
                                                     method='bilinear')
                elif self.ConfNet_resize == 'pad':
                    confidence_map = tf.image.resize_with_crop_or_pad(confidence_map_pred, img_shape[0], img_shape[1])

                confidence_map = tf.squeeze(confidence_map).numpy()

            elif self.module_name == 'LGC':
                disp = image_io.read(sample.disp_path) / 256.0
                loc = image_io.read(sample.local_path)
                glob = image_io.read(sample.global_path)

                disp = tf.expand_dims(disp, axis=0)
                loc = tf.expand_dims(loc, axis=0)
                glob = tf.expand_dims(glob, axis=0)

                disp = tf.expand_dims(disp, axis=-1)
                loc = tf.expand_dims(loc, axis=-1)
                glob = tf.expand_dims(glob, axis=-1)

                confidence_map = self.model.predict(x=(disp, loc, glob))[0, :, :, 0]

                confidence_map = tf.squeeze(confidence_map).numpy()

            elif self.module_name == 'LFN':

                disp = image_io.read(sample.disp_path)
                image = image_io.read(sample.left_image_path)

                disp = tf.expand_dims(disp, axis=0)
                image = tf.expand_dims(image, axis=0)

                disp = tf.expand_dims(disp, axis=-1)

                confidence_map = self.model.predict(x=(disp, image))[0, :, :, 0]

                confidence_map = tf.squeeze(confidence_map).numpy()

            if self.loss_type == 'Probabilistic':
                confidence_map = np.exp(confidence_map)

            # path=sample.result_path
            if not os.path.exists(sample.result_path):
                os.makedirs(sample.result_path)

            png_creation_text = ""
            current_image = os.path.basename(os.path.normpath(sample.result_path))
            # Save confidence map
            if self.save_png:
                confidence_map_visual = confidence_map

                if self.loss_type == 'Binary_Cross_Entropy':
                    confidence_map_visual = confidence_map_visual * 255

                confidence_map_visual = confidence_map_visual.astype(int)

                export_image_path = os.path.join(sample.result_path,
                                                 'ConfMap_' + self.module_name + '_' + current_image +
                                                 '.png')

                image_io.write(export_image_path, confidence_map_visual)
                png_creation_text = ",ConfMap.png"

            path = os.path.join(sample.result_path, 'ConfMap_' + self.module_name + '.pfm')
            image_io.write(path, confidence_map.astype(np.float32))

            disp_creation_text = ""
            # Save disparity map
            if self.save_disp_map:
                if self.module_name == 'CVA':
                    disp_map = np.argmin(cv.get_data(), 2)
                    image_io.write(sample.result_path + 'DispMap_' + self.module_name + '.png', disp_map)
                    disp_creation_text = ",DispMap.png"

            print("Created Confmap.pfm", png_creation_text, disp_creation_text, "for", self.module_name, "at",
                  sample.result_path)
