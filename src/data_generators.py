from math import floor
import random
from abc import abstractmethod

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence

import metrics
import cost_volume
import census_metric

# import from /utils
import utils.image_io as image_io

from platform import system

if system() == 'Windows':
    import pickle
else:
    import pickle5 as pickle


# import matplotlib.pyplot as plt


class DataSample:
    def __init__(self, isTrainingSet, gt_path, disp_path='', local_path='', global_path='', cost_volume_path='',
                 left_image_path='',
                 right_image_path='',
                 offset=[0, 0],
                 step_size=[1, 1], cost_volume_depth=256, result_path=''):
        self.isTrainingSet = isTrainingSet
        self.gt_path = gt_path
        self.cost_volume_path = cost_volume_path
        self.left_image_path = left_image_path
        self.right_image_path = right_image_path
        self.offset = offset
        self.step_size = step_size
        self.cost_volume_depth = cost_volume_depth
        self.result_path = result_path
        self.disp_path = disp_path
        self.local_path = local_path
        self.global_path = global_path


class TrainingSample:
    def __init__(self, sample_name, row, col, gt_value, isTrainingSet):
        self.sample_name = sample_name
        self.row = row
        self.col = col
        self.gt_value = gt_value
        self.isTrainingSet = isTrainingSet


class IDataGenerator(Sequence):
    """ Abstract base class for generating batches of data. """

    def __init__(self, model, data_samples, batch_size, dim, shuffle, augment, crop_width, crop_height, use_warp,
                 mode='extract'):

        # Set member variables
        self.use_warp = use_warp
        self.model = model
        self.mode = mode
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = 1
        self.shuffle = shuffle
        self.augment = augment
        self.disp_norm = 256.0
        self.crop_width = crop_width
        self.crop_height = crop_height

        self.training_samples = self.create_training_samples(data_samples)

        self.indexes = np.arange(len(self.training_samples))

        # Shuffle sample list for initialisation
        if self.shuffle == True:
            self.shuffle_training_samples()

    @abstractmethod
    def create_training_samples(self, data_samples):
        """ Create sample IDs based on the provided file list

        @warning This is an abstract function that has to be implemented in any inherited class.

        @param data_samples: List containing data samples.
        @return: A list containing training samples.
        """
        raise NotImplementedError

    def get_number_of_samples(self):
        positives = 0
        overall = 0

        for sample in self.training_samples:
            cv_extract, resize_factor = self.get_cv_extract(sample)
            disp_est = cv_extract[int((self.dim[0] - 1) / 2), int((self.dim[1] - 1) / 2), :].argmin()
            disp_est = tf.convert_to_tensor((1.0 / resize_factor) * disp_est, dtype=tf.float32)
            disp_gt = tf.convert_to_tensor(sample.gt_value, dtype=tf.float32)
            positives += tf.dtypes.cast(metrics.compute_labels(disp_est, disp_gt), dtype=tf.float32)
            overall += 1.0

        return overall, positives

    def get_positive_weight(self, overall, positives, print_numbers=False):
        negatives = overall - positives
        if print_numbers:
            print('===========================')
            print('Number of training samples:')
            print('   Overall: ' + str(overall))
            print('   Positives: ' + str(positives))
            print('   Negatives: ' + str(negatives))
            print('===========================')
        return negatives / positives

    def __data_generation__(self, training_samples):
        """ Creates a batch of data with reference lables based on the specified IDs.

        @warning This is an abstract function that has to be implemented in any inherited class

        @param training_samples: List containing training samples which should be used to create this batch.
        @return: A batch of data with corresponding labels.
        """

        if self.model == "CVA":
            # Batch initialisation

            X = np.empty((self.batch_size, *self.dim, self.n_channels))
            Y = np.empty((self.batch_size, 2), dtype=float)

            # Generate data
            for i, training_sample in enumerate(training_samples):
                # ID structure: cost volume path, row, column, gt_disp
                cv_extract, resize_factor = self.get_cv_extract(training_sample)

                # Generate disparity estimation and label
                disp_est = cv_extract[int((self.dim[0] - 1) / 2), int((self.dim[1] - 1) / 2), :].argmin()

                X[i, :, :, :, 0] = cv_extract[:, :, :]
                Y[i, 0] = (1.0 / resize_factor) * disp_est
                Y[i, 1] = training_sample.gt_value

            return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(Y, dtype=tf.float32)

        elif self.model == "ConfNet":
            if training_samples[0].isTrainingSet:

                # Batch initialisation according to chosen fusion model
                if self.use_warp == "EF":
                    X_image = np.empty((self.batch_size, *self.dim, 4))
                else:
                    X_image = np.empty((self.batch_size, *self.dim, 3))

                X_disp = np.empty((self.batch_size, *self.dim, 1))
                X_warp_errormap = np.empty((self.batch_size, *self.dim, 1))
                Y = np.empty((self.batch_size, *self.dim, 2))

                # Training set
                for i, training_sample in enumerate(training_samples):
                    image = self.image_dict[training_sample.sample_name]
                    disp = self.disp_dict[training_sample.sample_name]

                    # Data generation according to chosen fusion model
                    if self.use_warp == "EF":
                        warped_left = self.warped_errormap_dict[training_sample.sample_name]
                        image = tf.concat((image, tf.expand_dims(warped_left, axis=-1)), axis=-1)

                    if self.use_warp == "LF":
                        crop = tf.cast(self.get_crops(image, disp, self.gt_dict[training_sample.sample_name],
                                                      self.warped_errormap_dict[training_sample.sample_name]),
                                       dtype=tf.float32)
                        image_extract, disp_extract, gt_extract, em_extract = tf.split(crop,
                                                                                       [np.shape(X_image)[-1], 1, 1, 1],
                                                                                       axis=-1)
                        X_warp_errormap[i, :, :, :] = em_extract[:, :, :] / 256.0
                    else:
                        crops = tf.cast(self.get_crops(image, disp, self.gt_dict[training_sample.sample_name]),
                                        dtype=tf.float32)
                        image_extract, disp_extract, gt_extract = tf.split(crops, [np.shape(X_image)[-1], 1, 1], axis=2)

                    X_image[i, :, :, :] = image_extract[:, :, :] / 256.0
                    X_disp[i, :, :, :] = disp_extract[:, :, :] / 256.0
                    Y[i, :, :, 0] = disp_extract[:, :, 0]
                    Y[i, :, :, 1] = gt_extract[:, :, 0]
            else:
                # Validation set
                for i, training_sample in enumerate(training_samples):
                    disp = tf.squeeze(self.disp_dict[training_sample.sample_name], 0)
                    gt = tf.squeeze(self.gt_dict[training_sample.sample_name], 0)

                    if self.use_warp == "EF":
                        image = self.image_dict[training_sample.sample_name]
                        warp_errormap = self.warped_errormap_dict[training_sample.sample_name]
                        image = tf.cast(tf.concat((image, tf.expand_dims(warp_errormap, axis=-1)), axis=-1), tf.float32)
                    else:
                        image = tf.cast(self.image_dict[training_sample.sample_name], tf.float32)
                        warp_errormap = self.warped_errormap_dict[training_sample.sample_name]

                    # set shape as factor of 16 to remove padding
                    shape = tf.shape(image)
                    image_height = floor(shape[1] / 16) * 16
                    image_width = floor(shape[2] / 16) * 16

                    X_image = np.empty((self.batch_size, image_height, image_width, tf.shape(image)[-1]))

                    X_warp_errormap = np.empty((self.batch_size, image_height, image_width, 1))
                    if self.use_warp == "LF":
                        X_warp_errormap[i, :, :, :] = tf.cast(
                            tf.image.resize_with_crop_or_pad(tf.expand_dims(warp_errormap, axis=-1), image_height,
                                                             image_width), tf.float32) / 128.0 - 1

                    X_disp = np.empty((self.batch_size, image_height, image_width, 1))
                    Y = np.empty((self.batch_size, image_height, image_width, 2))

                    X_image[i, :, :, :] = tf.image.resize_with_crop_or_pad(image, image_height, image_width) / 256.0
                    X_disp[i, :, :, :] = tf.image.resize_with_crop_or_pad(disp, image_height, image_width) / 256.0
                    Y[i, :, :, 0] = tf.image.resize_with_crop_or_pad(disp, image_height, image_width)[:, :, 0]
                    Y[i, :, :, 1] = tf.image.resize_with_crop_or_pad(gt, image_height, image_width)[:, :, 0]

            return [tf.convert_to_tensor(X_image, dtype=tf.float32),
                    tf.convert_to_tensor(X_disp, dtype=tf.float32),
                    tf.convert_to_tensor(X_warp_errormap, dtype=tf.float32)], tf.convert_to_tensor(Y, dtype=tf.float32)

        elif self.model == "LGC":
            # Batch initialisation
            X_disp = np.empty((self.batch_size, *self.dim, 1)) / self.disp_norm
            X_loc = np.empty((self.batch_size, *self.dim, 1))
            X_glob = np.empty((self.batch_size, *self.dim, 1))
            Y = np.empty((self.batch_size, 2), dtype=float)

            # Generate data
            for i, training_sample in enumerate(training_samples):
                row = training_sample.row
                col = training_sample.col

                disp_extract = self.get_patch(self.disp_dict[training_sample.sample_name], row, col)
                loc_extract = self.get_patch(self.loc_dict[training_sample.sample_name], row, col)
                glob_extract = self.get_patch(self.glob_dict[training_sample.sample_name], row, col)

                X_disp[i, :, :, 0] = disp_extract / 256.0
                X_loc[i, :, :, 0] = loc_extract
                X_glob[i, :, :, 0] = glob_extract

                disp = self.disp_dict[training_sample.sample_name][0][row][col]

                Y[i, 0] = disp
                Y[i, 1] = training_sample.gt_value

            return [tf.convert_to_tensor(X_disp, dtype=tf.float32), tf.convert_to_tensor(X_loc, dtype=tf.float32),
                    tf.convert_to_tensor(X_glob, dtype=tf.float32)], tf.convert_to_tensor(Y, dtype=tf.float32)

        elif self.model == "LFN":
            # Batch initialisation
            X_disp = np.empty((self.batch_size, *self.dim, 1))
            X_image = np.empty((self.batch_size, *self.dim, 3))
            Y = np.empty((self.batch_size, 2), dtype=float)

            # Generate data
            for i, training_sample in enumerate(training_samples):
                row = training_sample.row
                col = training_sample.col

                disp_extract = self.get_patch(self.disp_dict[training_sample.sample_name], row, col)
                image_extract = self.get_patch3d(self.image_dict[training_sample.sample_name], row, col)

                X_disp[i, :, :, 0] = disp_extract
                X_image[i, :, :, :] = image_extract

                disp = self.disp_dict[training_sample.sample_name][0][row][col]
                Y[i, 0] = disp
                Y[i, 1] = training_sample.gt_value

            return [tf.convert_to_tensor(X_disp, dtype=tf.float32),
                    tf.convert_to_tensor(X_image, dtype=tf.float32)], tf.convert_to_tensor(Y, dtype=tf.float32)

    def get_crops(self, image, disp, gt, errormap=''):
        """ Extracts crop from 2d inputs.
        @param input: single input image, disparity map, groundtruth and errormap.
        @return: A crop array.
        """

        # Formatting for tf and concat input, assuring similar dim
        if errormap:
            errormap = np.expand_dims(errormap, axis=-1)
            concat = tf.concat((image, disp, gt, errormap), axis=-1)
        else:
            concat = tf.concat((image, disp, gt), axis=-1)

        # extract crop
        crops = tf.image.random_crop(tf.squeeze(concat), [self.crop_height, self.crop_width, concat.shape[-1]])

        return crops

    @staticmethod
    def get_patch(img_list, row, col, patch_size=9):
        """ Extracts patch from 2d input.
        @param: List containing images with row and col of available groundtruth and patchsize.
        @return: Image excerpt of patchsize at groundtruth.
        """
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

    @staticmethod
    def get_patch3d(img_list, row, col, patch_size=9):
        """ Extracts patch from 3d input.
        @param input: single input image, disparity map, groundtruth and errormap.
        @return: image excerpt.
        """
        img = img_list[0]

        excerpt = np.zeros((patch_size, patch_size, 3))
        nb_offset = ((patch_size - 1) / 2)

        for excerpt_row in range(0, patch_size):
            for excerpt_col in range(0, patch_size):
                image_row = int(row + excerpt_row - nb_offset)
                image_col = int(col + excerpt_col - nb_offset)

                if 0 <= image_row <= (img.shape[0] - 1) and 0 <= image_col <= (img.shape[1] - 1):
                    excerpt[excerpt_row, excerpt_col, 0] = img[image_row, image_col, 0]
                    excerpt[excerpt_row, excerpt_col, 1] = img[image_row, image_col, 1]
                    excerpt[excerpt_row, excerpt_col, 2] = img[image_row, image_col, 2]

        return excerpt

    @abstractmethod
    def get_cv_extract(self):
        raise NotImplementedError

    def __len__(self):
        """ Denotes the number of batches per epoch.

        @return: Number of batches per epoch.
        """
        return int(np.floor(len(self.training_samples) / self.batch_size))

    def __getitem__(self, index):
        """ Creates a batch of data with reference lables for a specified batch index.

        @param index: Index of the batch to be created
        @return: A batch of data with corresponding labels.
        """

        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        training_samples = [self.training_samples[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation__(training_samples)
        return X, y

    def shuffle_training_samples(self):
        """ Shuffles the list of batch indices. """
        self.indexes = np.arange(len(self.training_samples))
        np.random.shuffle(self.indexes)

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if (self.shuffle):
            self.shuffle_training_samples()

    def training_samples_from_GT(self, sample_name, gt_path, step_size, offset):
        """ Creates a set of sample IDs based on a specified reference disparty map.

        Based on the specified step size and offset, the reference disparity map is sampled and for every pixel
        with a reference disparity available one sample ID is created.

        @param sample_name: Name of the current sample (e.g. left image path, cost volume path)
        @param gt_path: Path of the reference disparity map
        @param step_size: Specifies the distance between two sample points
        @param offset: Specifies the offset of the first sample point from the image origin
        @return: A list of sample IDs and the normalised reference disparity map
        """

        # Read ground truth and normalise values if necessary
        disp_gt = image_io.read(gt_path)
        dimensions = disp_gt.shape

        # Assure that there are no constructs like -inf, inf
        disp_gt[disp_gt == -np.inf] = 0
        disp_gt[disp_gt == np.inf] = 0

        # Check for available ground truth points
        training_samples = []
        for row in range(offset[0], dimensions[0], step_size[0]):
            for col in range(offset[1], dimensions[1], step_size[1]):
                gt_value = disp_gt[row][col]
                if (gt_value != 0):
                    # Ground truth point is available -> Create sample for this pixel
                    training_samples.append(TrainingSample(sample_name, row, col, gt_value, True))

        return training_samples, disp_gt


class DataGeneratorCV(IDataGenerator):
    """ Generates batches of data based on a previously computed cost volume """

    # @brief Initialises the data generator
    # @warning All specified cost volumes are loaded to memory before training can be started
    # @param data_file_paths One tuple per cost volume: 
    # [{cost volume path}, {gt image path}, gt_norm_factor, step_size, offset]
    def __init__(self, model='CVA', data_samples=[], batch_size=8, dim=(13, 13, 256), shuffle=False,
                 augment=False, cv_norm=[0.0, 1.0], crop_width='', crop_height='', use_warp=''):

        # Call constructor of abstract base class
        super(DataGeneratorCV, self).__init__(model, data_samples, batch_size, dim, shuffle, augment, crop_width,
                                              crop_height, use_warp)
        self.cv_norm = cv_norm

        # Load cost volumes
        self.cv_dict = self.create_cv_dict(data_samples)

    def create_training_samples(self, data_samples):
        """ Implementation of the abstract function defined in the base class. """

        training_samples = []

        # Iterate over the provided cost volumes
        for data_sample in data_samples:
            # Get datasamples and normalised ground truth for current sample
            curr_data_samples, _ = self.training_samples_from_GT(sample_name=data_sample.cost_volume_path,
                                                                 gt_path=data_sample.gt_path,
                                                                 step_size=data_sample.step_size,
                                                                 offset=data_sample.offset)
            training_samples.extend(curr_data_samples)

        return training_samples

    def create_cv_dict(self, data_samples):
        """ Loads all cost volumes to memory.

        @param data_samples: List of data samples to load.
        @return: A dictionary containing all specified cost volumes with their path as key attribute.
        """

        cv_dict = {}
        for data_sample in data_samples:

            # Load and store cost volume
            cv_path = data_sample.cost_volume_path
            cv = cost_volume.CostVolume()
            if cv_path[-3:] == 'bin':
                # To get the cost volume dimensions the ground truth disparity map is used
                disp_gt = image_io.read(data_sample.gt_path)
                cv.load_bin(cv_path, disp_gt.shape[0], disp_gt.shape[1], data_sample.cost_volume_depth)

            elif cv_path[-3:] == 'dat':
                cv.load_dat(cv_path)

            else:
                with open(cv_path, 'rb') as file:
                    cv = pickle.load(file)

            if data_sample.cost_volume_depth > self.dim[2]:
                cv.reduce_depth(self.dim[2])

            # Normalise the cost volume
            cv.normalise(self.cv_norm[0], self.cv_norm[1])
            cv_dict[cv_path] = cv
        return cv_dict

    def get_cv_extract(self, training_sample):
        """ Implementation of the abstract function defined in the base class. """
        return self.cv_dict[training_sample.sample_name].get_excerpt((training_sample.row, training_sample.col),
                                                                     self.dim[0]), 1.0


class DataGeneratorImage(IDataGenerator):
    """ Generates batches of data based on a stereo image pair using the Census metric. """

    # @brief Initialise the data generator
    # @param data_file_paths One tuple per image pair: 
    # [{left image path}, {right image path}, {gt image path}, gt_norm_factor, step_size, offset]
    def __init__(self, model='CVA', data_samples=[], batch_size=8, dim=(13, 13, 256), shuffle=False,
                 augment=False, cv_norm=[0.0, 1.0], crop_width='', crop_height=''):

        # Set member variables        
        self.decrease_prob = 0.5
        self.resize_factor_range = [0.25, 2.0]
        self.metric_filter_size = 5
        self.smooth_filter_size = 5
        self.metric = census_metric.CensusMetric(self.metric_filter_size, self.smooth_filter_size)
        self.resize_factor_dict = None

        # Call constructor of abstract base class
        super(DataGeneratorImage, self).__init__(model, data_samples, batch_size, dim, shuffle, augment, crop_width,
                                                 crop_height)

        # Load images
        self.image_dict = self.create_image_dict(data_samples)

        # Create Census transformations
        self.census_dict = self.create_census_dict()

    def create_training_samples(self, data_samples):
        """ Implementation of the abstract function defined in the base class. """

        training_samples = []
        resize_factor_dict = {}

        # Iterate over the provided image pairs
        for data_sample in data_samples:
            # Get datasamples and normalised ground truth for current sample
            sample_name = data_sample.left_image_path
            curr_data_samples, disp_gt = self.training_samples_from_GT(sample_name=sample_name,
                                                                       gt_path=data_sample.gt_path,
                                                                       step_size=data_sample.step_size,
                                                                       offset=data_sample.offset)
            training_samples.extend(curr_data_samples)

            # Get max disparity and determine the resize factor range of the specific image           
            max_resize_factor = self.dim[2] / disp_gt.max()
            max_resize_factor = min(max_resize_factor, self.resize_factor_range[1])
            min_resize_factor = min(max_resize_factor, self.resize_factor_range[0])
            resize_factor_dict[sample_name] = [min_resize_factor, max_resize_factor]

        self.resize_factor_dict = resize_factor_dict
        return training_samples

    def create_image_dict(self, data_samples):
        """ Loads images based on provided file list and stores them in a dictionary.

        @param data_samples: List containing data samples
        @return: A dictionary of type: key: left image path, data: tuple[left image, left census, right image, right census]
        """

        image_dict = {}
        for data_sample in data_samples:
            # Load and store left and right image
            left_image = image_io.read(data_sample.left_image_path)
            right_image = image_io.read(data_sample.right_image_path)
            image_dict[data_sample.left_image_path] = [left_image, right_image]
        return image_dict

    def create_census_dict(self):
        """ Loads images based on provided file list and stores them in a dictionary.

        @return: A dictionary of type: key: left image path, data: tuple[left census, right census, resize_factor]
        """

        census_dict = {}

        for name, images in self.image_dict.items():

            # Generate random resize factor
            resize_factor_range = self.resize_factor_dict[name]

            if (self.augment and resize_factor_range[0] != resize_factor_range[1]):

                if (random.uniform(0.0, 1.0) > self.decrease_prob and resize_factor_range[1] > 1.0):
                    # Increase image dimensions
                    resize_factor = round(random.uniform(1.0, resize_factor_range[1]), 2)
                else:
                    # Decrease image dimensions
                    resize_factor = round(random.uniform(resize_factor_range[0], min(resize_factor_range[1], 1.0)), 2)
            else:
                resize_factor = min(resize_factor_range[1], 1.0)

            left_image = images[0]
            right_image = images[1]

            # Resize images if requested
            if (resize_factor < 1.0):
                left_image = cv2.resize(left_image, None, fx=resize_factor, fy=resize_factor,
                                        interpolation=cv2.INTER_AREA)
                right_image = cv2.resize(right_image, None, fx=resize_factor, fy=resize_factor,
                                         interpolation=cv2.INTER_AREA)
            elif (resize_factor > 1.0):
                left_image = cv2.resize(left_image, None, fx=resize_factor, fy=resize_factor,
                                        interpolation=cv2.INTER_CUBIC)
                right_image = cv2.resize(right_image, None, fx=resize_factor, fy=resize_factor,
                                         interpolation=cv2.INTER_CUBIC)

            # Transform images
            census_dict[name] = [self.metric.__create_census_trafo__(left_image),
                                 self.metric.__create_census_trafo__(right_image), resize_factor]

        return census_dict

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if (self.shuffle):
            self.shuffle_training_samples()

        # Create new Census transformations with different resize factors 
        # after each epoch if augmentation is required
        if (self.augment):
            self.create_census_dict()

    def get_cv_extract(self, training_sample):
        """ Implementation of the abstract function defined in the base class. """
        census_trafos = self.census_dict[training_sample.sample_name]
        resize_factor = census_trafos[2]

        # Generate cost volume extract
        row = int(training_sample.row * resize_factor)
        col = int(training_sample.col * resize_factor)

        cv_extract = self.metric.__compute_cv_extract__(census_trafos[0], census_trafos[1],
                                                        row, col, self.dim)

        return cv_extract, resize_factor


class DataGeneratorConfNet(IDataGenerator):
    """ Loads images based on provided file list and stores them in a dictionary.

    @return: A dictionary of type: key: left image path, data: tuple[left, disparity, warped left, warped errormap, groundtruth]
    """

    def __init__(self, model='ConfNet', data_samples=[], batch_size=1, dim=(256, 512), shuffle=False,
                 augment=False, crop_width=256, crop_height=512, use_warp=''):
        self.crop_width = crop_width
        self.crop_height = crop_height

        # Call constructor of abstract base class
        super(DataGeneratorConfNet, self).__init__(model, data_samples, batch_size, dim, shuffle, augment, crop_width,
                                                   crop_height, use_warp)

        if self.shuffle:
            data_samples = random.sample(data_samples, len(data_samples))

        self.image_dict, self.disp_dict, self.gt_dict, self.left_warped_dict, self.warped_errormap_dict = self.create_dicts(
            data_samples)

    def create_dicts(self, data_samples):
        image_dict = {}
        disp_dict = {}
        gt_dict = {}
        left_warped_dict = {}
        warped_errormap_dict = {}

        for data_samples in data_samples:
            # Load and store image
            left = tf.cast(self.read_tf_image(data_samples.left_image_path, [None, None, 3]), tf.float32)
            right = tf.cast(self.read_tf_image(data_samples.right_image_path, [None, None, 3]), tf.float32)
            disp = tf.cast(self.read_tf_image(data_samples.disp_path, [None, None, 1], dtype=tf.uint16),
                           tf.float32) / 256.0
            gt = tf.cast(self.read_tf_image(data_samples.gt_path, [None, None, 1], dtype=tf.uint16), tf.float32) / 256.0

            # Reshape for warped transform
            rightRGB = tf.reverse(right, axis=[-1])
            leftRGB = tf.reverse(left, axis=[-1])

            # Form warped difference
            left_warped = image_io.to_warped_image(rightRGB.numpy(), np.squeeze(disp.numpy()), 'r2l')
            warped_errormap = image_io.get_warp_errormap(leftRGB.numpy(), left_warped)

            # Set dict for every input, using a collective key
            image_dict[data_samples.left_image_path] = [left]
            disp_dict[data_samples.left_image_path] = [disp]
            left_warped_dict[data_samples.left_image_path] = [left_warped]
            warped_errormap_dict[data_samples.left_image_path] = [warped_errormap]
            gt_dict[data_samples.left_image_path] = [gt]

        return image_dict, disp_dict, gt_dict, left_warped_dict, warped_errormap_dict

    def create_training_samples(self, data_samples):
        """ Implementation of the abstract function defined in the base class. """

        training_samples = []
        resize_factor_dict = {}

        # Iterate over the provided image pairs
        for data_sample in data_samples:
            # Get datasamples and normalised ground truth for current sample
            sample_name = data_sample.left_image_path

            training_samples.append(TrainingSample(sample_name, 0, 0, 0, data_sample.isTrainingSet))

        return training_samples

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            self.shuffle_training_samples()

    @staticmethod
    def read_tf_image(image_path, shape=None, dtype=tf.uint8):
        """ Read images in tf-format, while decoding if needed.
        @return: image
        """

        image_raw = tf.io.read_file(image_path)
        if dtype == tf.uint8:
            image = tf.image.decode_image(image_raw)
        else:
            image = tf.image.decode_png(image_raw, dtype=dtype)

        if shape is None:
            image.set_shape([None, None, 3])
        else:
            image.set_shape(shape)

        return image


class DataGeneratorLGC(IDataGenerator):
    """ Loads images based on provided file list and stores them in a dictionary.

    @return: A dictionary of type: key: left image path, data: tuple[disparity, confidence map (local and global), groundtruth]
    """

    def __init__(self, model='LGC', data_samples=[], batch_size=64, dim=(9, 9), shuffle=False,
                 augment=False, crop_width='', crop_height='', use_warp=''):

        super(DataGeneratorLGC, self).__init__(model, data_samples, batch_size, dim, shuffle, augment, crop_width,
                                               crop_height, use_warp)

        self.disp_dict, self.loc_dict, self.glob_dict, self.gt_dict = self.create_dicts(data_samples)

    def create_dicts(self, data_samples):
        disp_dict = {}
        loc_dict = {}
        glob_dict = {}
        gt_dict = {}

        for data_samples in data_samples:
            disp = image_io.read(data_samples.disp_path)
            loc = image_io.read(data_samples.local_path)
            glob = image_io.read(data_samples.global_path)
            gt = image_io.read(data_samples.gt_path)

            disp_dict[data_samples.disp_path] = [disp]
            loc_dict[data_samples.disp_path] = [loc]
            glob_dict[data_samples.disp_path] = [glob]
            gt_dict[data_samples.disp_path] = [gt]

        return disp_dict, loc_dict, glob_dict, gt_dict

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            self.shuffle_training_samples()

    def create_training_samples(self, data_samples):
        """ Implementation of the abstract function defined in the base class. """

        training_samples = []

        # Iterate over the provided image pairs
        for data_sample in data_samples:
            # Get datasamples and normalised ground truth for current sample
            sample_name = data_sample.disp_path
            curr_data_samples, disp_gt = self.training_samples_from_GT(sample_name=sample_name,
                                                                       gt_path=data_sample.gt_path,
                                                                       step_size=data_sample.step_size,
                                                                       offset=data_sample.offset)
            training_samples.extend(curr_data_samples)

        return training_samples


class DataGeneratorLFN(IDataGenerator):
    """ Loads images based on provided file list and stores them in a dictionary.

    @return: A dictionary of type: key: left image path, data: tuple[left image, disparity, groundtruth]
    """
    def __init__(self, model='LFN', data_samples=[], batch_size=128, dim=(9, 9), shuffle=False,
                 augment=False, crop_width='', crop_height='', use_warp=''):

        super(DataGeneratorLFN, self).__init__(model, data_samples, batch_size, dim, shuffle, augment, crop_width,
                                               crop_height, use_warp)

        self.disp_dict, self.image_dict, self.gt_dict = self.create_dicts(data_samples)

    def create_dicts(self, data_samples):
        disp_dict = {}
        image_dict = {}
        gt_dict = {}

        for data_samples in data_samples:
            disp = image_io.read(data_samples.disp_path)
            image = image_io.read(data_samples.left_image_path)
            gt = image_io.read(data_samples.gt_path)

            disp_dict[data_samples.disp_path] = [disp]
            image_dict[data_samples.disp_path] = [image]
            gt_dict[data_samples.disp_path] = [gt]

        return disp_dict, image_dict, gt_dict

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            self.shuffle_training_samples()

    def create_training_samples(self, data_samples):
        """ Implementation of the abstract function defined in the base class. """

        training_samples = []

        # Iterate over the provided image pairs
        for data_sample in data_samples:
            # Get datasamples and normalised ground truth for current sample
            sample_name = data_sample.disp_path
            curr_data_samples, disp_gt = self.training_samples_from_GT(sample_name=sample_name,
                                                                       gt_path=data_sample.gt_path,
                                                                       step_size=data_sample.step_size,
                                                                       offset=data_sample.offset)
            training_samples.extend(curr_data_samples)

        return training_samples
