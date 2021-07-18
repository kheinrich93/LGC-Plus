import numpy as np
import cv2

import os
import glob

#import from /utils
import utils.image_io

def normalise_image(image, src_min=0.0, src_max=255.0, dest_min=-1.0, dest_max=1.0):
    """ Transforms the values within an image from a specified source range to a specified destination range.

    @param image: Numpy array representing an image
    @param src_min: Minimum value of the source range
    @param src_max: Maximum value of the source range
    @param dest_min: Minimum value of the destination range
    @param dest_max: Maximum value of the destination range
    @return: The transformed image
    """
    scale_factor = (src_max - src_min) / (dest_max - dest_min)
    normalised_image = (image - src_min) * (1.0 / scale_factor) + dest_min
    return normalised_image


def resize_image(image, resize_factor, divisibility=1):
    """ Resizes an image to a format which is suitable for a specific network architecture.

    @param image: Input image.
    @param resize_factor: Factor which is applied to the image first in order to change it size.
    @param divisibility: The image is shrunk independently in x- and y- direction so that it's dimensions are dividable
    by the specified factor.
    @return: The resized image, and the scale factors in x- and y- direction that were applied in total.
    """
    orig_size = image.shape
    dest_x = resize_factor * orig_size[0]
    dest_y = resize_factor * orig_size[1]
    diff_x = dest_x % divisibility
    diff_y = dest_y % divisibility
    scale_factor_height = (dest_x - diff_x) / orig_size[0]
    scale_factor_width = (dest_y - diff_y) / orig_size[1]
    resized_image = cv2.resize(image, (0, 0), fy=scale_factor_height, fx=scale_factor_width,
                               interpolation=cv2.INTER_AREA)
    return resized_image, scale_factor_height, scale_factor_width


def save_disparity_map(image, file_path, scale_factor_height=1.0, scale_factor_width=1.0, save_png=True,
                       file_extension='.pfm'):
    """ Saves a disparity map to file.

    @param image: Image to be written to file.
    @param file_path: Path of the resulting file.
    @param scale_factor_height: Height factor that was applied to shrink the input images.
    @param scale_factor_width: Width factor that was applied to shrink the input images.
    @param save_png: Specifies if the disparity map should additionally be saved in png format with rounded values.
    @param file_extension: Specifies the file extension used to save the results
    """
    image = cv2.resize(image, (0, 0), fy=(1.0 / scale_factor_height), fx=(1.0 / scale_factor_width))
    image = image * (1.0 / scale_factor_width)
    image_io.write(file_path + file_extension, image.astype(np.float32))

    if save_png:
        image_io.write(file_path + '.png', disp_to_img(image))


def disp_to_img(mat):
    sample = np.round(mat)
    return sample.astype(np.uint8)


# TODO: Fuse metric definitions with the ones in metrics.py to avoid double definitions
def mae(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    diff_nz = diff[y_true.astype(dtype=bool)]
    return np.mean(diff_nz)


def rmse(y_true, y_pred):
    sqr_diff = (y_true - y_pred) ** 2
    sqr_diff_nz = sqr_diff[y_true.astype(dtype=bool)]
    mean = np.mean(sqr_diff_nz)
    return np.sqrt(mean)


def pixel_error(y_true, y_pred, threshold):
    diff = np.abs(y_true - y_pred)
    diff_nz = diff[y_true.astype(dtype=bool)]
    return (np.count_nonzero(np.greater(diff_nz, threshold)) / diff_nz.size)


def delete_summary_images():
    logdir = "logs/train_data/"
    files = glob.glob(logdir + '/*.v2')

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    if files == []:
        print("No tf-images to delete")
    else:
        print("Successfully deleted tf-images")
