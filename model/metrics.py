import math
import numpy as np
import tensorflow as tf

def compute_labels(disp_est, disp_gt, treshold_abs=3.0, threshold_rel=0.05):
    diff = tf.abs(disp_est - disp_gt)
    label_abs = tf.greater(diff, tf.constant([treshold_abs]))
    label_rel = tf.greater(diff, tf.constant([threshold_rel]) * tf.abs(disp_gt))
    return tf.math.logical_not(tf.math.logical_and(label_abs, label_rel))


class Metrics:
    """ Contains and manages loss functions and further metrics for training CVA-MMNet based networks.

    If the generic loss function is used, the Metrics object takes care of the correct configuration according to the
    specified member variables.
    """

    def __init__(self, basic_loss, pos_class_weight=1.0):
        """ Initialisation for the metrics class.

        @param basic_loss: String that specifies which kind of loss function should be used.
        @param pos_class_weight: Weighting of positive samples, while negative samples are weightes with 1.0.
                                 (Only used for classification-based losses.)
        This information is of special importance for the additional metrics since their computation may
        vary based on the utilised loss function.
        """
        self.basic_loss = basic_loss
        self.pos_class_weight = pos_class_weight

    @staticmethod
    def export_tf_image(label, pred, gt):
        # extract images after every epoch
        # can be watched with:~ tensorboard --logdir logs/train_data
        # at localhost:6006

        error_map = np.zeros(tf.shape(label))
        np.subtract(tf.squeeze(label), tf.squeeze(pred), out=error_map, where=gt > 0.1)
        error_map = np.absolute(tf.expand_dims(error_map, -1))

        # Sets up a timestamped log directory.
        logdir = "logs/train_data/"

        # Creates a file writer for the log directory.
        file_writer = tf.summary.create_file_writer(logdir)

        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image("Training data", error_map, step=0)


    @staticmethod
    def binary_crossentropy(y_true, y_pred, model_name, epoch):

        if model_name == "ConfNet":
            disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=3)
            disp_est = tf.squeeze(disp_est, axis=[3])
            disp_gt = tf.squeeze(disp_gt, axis=[3])
            label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)

            # tf.print("y_pred:", y_pred.shape)

            # Compute pixel-wise binary cross-entropy
            label_gt = tf.stack([label_gt, 1.0 - label_gt], axis=3)
            y_pred = tf.stack([y_pred, 1.0 - y_pred], axis=3)

            # tf.print("label_gt:", label_gt.shape)
            # tf.print("y_pred stack:", y_pred.shape)

            y_pred = tf.squeeze(y_pred, axis=[4])
            # label_gt = tf.squeeze(label_gt)

            if epoch+1 % 10 == 0:
                Metrics.export_tf_image(label_gt[:, :, :, 0], y_pred[:, :, :, 0], disp_gt)

            # tf.print("y_pred squeeze:", y_pred.shape)
            # tf.print("label_gt squeeze:", label_gt.shape)

            bce = tf.keras.losses.binary_crossentropy(label_gt, y_pred)

            # tf.print("bce:", bce.shape)
            # disp_gt = tf.squeeze(disp_gt)
            # tf.print("disp_gt: ", disp_gt.shape)

            # Filter for pixels with available ground truth
            filtered_bce = tf.boolean_mask(bce, disp_gt)

            # Mean binary cross-entropy
            if len(filtered_bce) == 0:
                return tf.constant(0.0)
            else:
                return tf.math.reduce_mean(filtered_bce)

        elif model_name == "CVA":

            disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
            label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)

            return tf.keras.losses.binary_crossentropy(tf.squeeze(label_gt), tf.squeeze(y_pred))

        elif model_name == "LGC":
            disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
            label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)
            return tf.keras.losses.binary_crossentropy(tf.squeeze(label_gt), tf.squeeze(y_pred))

        elif model_name == "LFN":
            disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
            label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)
            return tf.keras.losses.binary_crossentropy(tf.squeeze(label_gt), tf.squeeze(y_pred))

    def weighted_binary_crossentropy(self, y_true, y_pred):

        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        label_gt = tf.squeeze(tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32))
        weights = (label_gt * (self.pos_class_weight - 1.0)) + 1.0

        y_pred = tf.squeeze(y_pred)

        label_gt = tf.stack([label_gt, 1.0 - label_gt], axis=1)
        y_pred = tf.stack([y_pred, 1.0 - y_pred], axis=1)

        # bce = -label_gt * tf.math.log(y_pred) - (1.0 - label_gt) * tf.math.log(1.0 - y_pred)

        # bce = tf.keras.losses.BinaryCrossentropy(
        #    from_logits=False, label_smoothing=0, reduction=tf.keras.losses.Reduction.NONE)
        losses = tf.keras.losses.binary_crossentropy(label_gt, y_pred)

        # tf.print(tf.shape(label_gt))
        # tf.print(tf.shape(y_pred))
        # tf.print(tf.shape(losses))

        weighted_bce = weights * losses
        mean_bce = tf.math.reduce_mean(weighted_bce)

        '''
        if math.isnan(mean_bce):
            tf.print('y_true: ', y_true, summarize=-1)
            tf.print('y_pred: ', y_pred, summarize=-1)
            tf.print('Est: ', disp_est, summarize=-1)
            tf.print('GT: ', disp_gt, summarize=-1)
            tf.print('label_gt: ', label_gt, summarize=-1)
            tf.print('weights: ', weights, summarize=-1)
            tf.print('bce: ', losses, summarize=-1)
            tf.print('weighted_bce: ', weighted_bce, summarize=-1)
            tf.print('mean_bce: ', mean_bce, summarize=-1)
            raise Exception('Problem with loss!')
        '''
        return mean_bce

    def accuracy(self, y_true, y_pred, model_name,epoch):
        """ Computes the binary accuracy if uncertainty prediction is interpreted as classification task.

        @param y_true: Tensor containing estimated and reference disparity.
        @param y_pred: The predicted label.
        @return: The accuracy of the estimated binary labels.
        """
        if model_name == "ConfNet":
            disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=3)
            disp_est = tf.squeeze(disp_est, axis=[3])
            disp_gt = tf.squeeze(disp_gt, axis=[3])

            label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)

            label_diff = tf.math.abs(label_gt - tf.squeeze(y_pred))
            label_diff = tf.boolean_mask(label_diff, disp_gt)

            label_diff = 1.0 - np.round(label_diff, 0)

            if len(label_diff) == 0:
                return tf.constant(0.0)
            else:
                return tf.constant(np.mean(label_diff))


        elif model_name == "CVA":
            disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
            label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)
            return tf.metrics.binary_accuracy(tf.squeeze(label_gt), tf.squeeze(y_pred))

        elif model_name == "LGC":
            disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
            label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)
            return tf.metrics.binary_accuracy(tf.squeeze(label_gt), tf.squeeze(y_pred))

        elif model_name == "LFN":
            disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
            label_gt = tf.dtypes.cast(compute_labels(disp_est, disp_gt), dtype=tf.float32)
            return tf.metrics.binary_accuracy(tf.squeeze(label_gt), tf.squeeze(y_pred))

    def residual_loss(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        loss = tf.abs(tf.squeeze(disp_est) + tf.squeeze(y_pred) - tf.squeeze(disp_gt))
        return tf.math.reduce_mean(loss)

    def residual_loss_abs(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        loss = tf.abs(tf.abs(tf.squeeze(disp_est) - tf.squeeze(disp_gt)) - tf.abs(tf.squeeze(y_pred)))
        return tf.math.reduce_mean(loss)

    def probabilistic_loss(self, y_true, y_pred):
        disp_est, disp_gt = tf.split(y_true, num_or_size_splits=2, axis=1)
        loss = (math.sqrt(2) * tf.math.abs(tf.squeeze(disp_gt) - tf.squeeze(disp_est)) *
                tf.math.exp(-tf.squeeze(y_pred))) + tf.squeeze(y_pred)
        return tf.math.reduce_mean(loss)

    def generic_loss(self):
        """ Wrapper for a generic loss functions, which computes the loss based on its Metrics object configuration.

        @return: Float value representing the computed loss.
        """

        def generic_loss_wrapped(y_true, y_pred, model_name, epoch):
            if self.basic_loss == 'Binary_Cross_Entropy':
                # loss = self.weighted_binary_crossentropy(y_true, y_pred)
                loss = self.binary_crossentropy(y_true, y_pred, model_name, epoch)
            elif self.basic_loss == 'Probabilistic':
                loss = self.probabilistic_loss(y_true, y_pred)
            elif self.basic_loss == 'Residual':
                loss = self.residual_loss(y_true, y_pred)
            elif self.basic_loss == 'Residual_Abs':
                loss = self.residual_loss_abs(y_true, y_pred)
            else:
                raise Exception('Unknown loss type: %s' % self.basic_loss)

            return loss

        return generic_loss_wrapped
