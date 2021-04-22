import time
from symbol import parameters
import tensorflow as tf

import data_generators
import graph
import params
import metrics

#import from /utils
from utils.training_loop import ITrainingLoop


class Train(ITrainingLoop):
    """ Implementation of a custom training loop used to train different versions of MMNet. """

    def __init__(self, parameter=params.Params(), network_name="MMNet_{0}".format(int(time.time())),
                 root_dir='experiments', experiment_series='', pretrained_network=''):

        """ Default initialisation function.

        @param parameter: An parameter object containing all relevant training parameters.
        @param network_name: The name of the network to be trained.
        @param root_dir: Root directory used for every output produced during training.
        @param experiment_series: Path relative to the root dir used for every output produced during training.
        @param pretrained_network: Path to a file containing pretrained model weights.
        """
        super(Train, self).__init__(parameter=parameter, network_name=network_name, root_dir=root_dir,
                                    experiment_series=experiment_series, pretrained_network=pretrained_network)

        # ---------------------------
        # Initialise optimiser and metrics
        # ---------------------------
        self.metrics_object = metrics.Metrics(basic_loss=parameter.loss_type,
                                              pos_class_weight=parameter.pos_class_weight)
        self.optimizer = tf.keras.optimizers.Adam(lr=parameter.learning_rate)

        self.model_name = parameter.module_name

        if parameter.task_type == 'Classification':
            self.metric_names = ['Loss', 'Accuracy']
            metric_formats = ['{:.3f}', '{:.3f}']
        else:
            self.metric_names = ['Loss']
            metric_formats = ['{:.3f}']
        self.metric_tracking.add_metrics(self.metric_names, metric_formats)

    def setup_model(self, parameters):
        """ Implementation of the abstract function defined in the base class. """
        if parameters.module_name == 'CVA':
            modelGraph = graph.CVANet().get_model(parameters)
            return modelGraph

        elif parameters.module_name == 'ConfNet':
            modelGraph = graph.ConfNet().get_model(parameters)
            return modelGraph

        elif parameters.module_name == 'LGC':
            modelGraph = graph.LGC().get_model(parameters)
            return modelGraph

        elif parameters.module_name == 'LFN':
            modelGraph = graph.LFN().get_model(parameters)
            return modelGraph
        else:
            raise Exception('Cannot setup graph for: ' + parameters.module_name)

        # Plot model
        if parameters.plot_graph:
           dot_img_file = 'D:/Master/masterthesis/MMNet/CVA-MMNet/'+parameters.model_name+'.png'
           tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)

    def setup_generators(self, parameters):
        """ Implementation of the abstract function defined in the base class. """
        sample_dims = (parameters.nb_size, parameters.nb_size, parameters.cost_volume_depth)

        if parameters.module_name == 'CVA':
            if parameters.CVA_data_mode == 'cv':
                training_generator = data_generators.DataGeneratorCV(data_samples=parameters.training_data,
                                                                     batch_size=parameters.batch_size, dim=sample_dims,
                                                                     shuffle=True, augment=False,
                                                                     cv_norm=parameters.cv_norm)
                validation_generator = data_generators.DataGeneratorCV(data_samples=parameters.validation_data,
                                                                       batch_size=parameters.batch_size,
                                                                       dim=sample_dims,
                                                                       shuffle=False, augment=False,
                                                                       cv_norm=parameters.cv_norm)
            elif parameters.CVA_data_mode == 'image':
                training_generator = data_generators.DataGeneratorImage(data_samples=parameters.training_data,
                                                                        batch_size=parameters.batch_size,
                                                                        dim=sample_dims,
                                                                        shuffle=True, augment=False)
                validation_generator = data_generators.DataGeneratorImage(data_samples=parameters.validation_data,
                                                                          batch_size=parameters.batch_size,
                                                                          dim=sample_dims,
                                                                          shuffle=False, augment=False)
            else:
                raise Exception('Unknown data_mode: ' + parameters.CVA_data_mode)
        # added
        # ConfNet image, disp
        elif parameters.module_name == 'ConfNet':
            training_generator = data_generators.DataGeneratorConfNet(data_samples=parameters.training_data,
                                                                      batch_size=parameters.batch_size,
                                                                      dim=(
                                                                          parameters.crop_height,
                                                                          parameters.crop_width),
                                                                      crop_width=parameters.crop_width,
                                                                      crop_height=parameters.crop_height,
                                                                      shuffle=True, use_warp=parameters.use_warp
                                                                      )
            # validation not on crops
            validation_generator = data_generators.DataGeneratorConfNet(data_samples=parameters.validation_data,
                                                                        batch_size=parameters.batch_size,
                                                                        dim=(
                                                                            parameters.crop_width,
                                                                            parameters.crop_height),
                                                                        crop_width=parameters.crop_width,
                                                                        crop_height=parameters.crop_height,
                                                                        shuffle=False, use_warp=parameters.use_warp
                                                                        )
        # For LGC input changes
        # Load precomputed loc and glob conf, disp, patches
        elif parameters.module_name == 'LGC':
            training_generator = data_generators.DataGeneratorLGC(data_samples=parameters.training_data,
                                                                  batch_size=parameters.batch_size,
                                                                  shuffle=True)

            validation_generator = data_generators.DataGeneratorLGC(data_samples=parameters.validation_data,
                                                                    batch_size=parameters.batch_size,
                                                                    shuffle=False)

        elif parameters.module_name == 'LFN':
            training_generator = data_generators.DataGeneratorLFN(data_samples=parameters.training_data,
                                                                  batch_size=parameters.batch_size,
                                                                  shuffle=True)

            validation_generator = data_generators.DataGeneratorLFN(data_samples=parameters.validation_data,
                                                                    batch_size=parameters.batch_size,
                                                                    shuffle=False)

        else:
            raise Exception('Unknown module: ' + parameters.module_name, '. Choose: CVA, ConfNet or LGC.')

        return training_generator, validation_generator

    def train_on_batch(self, X, y, epoch):
        """ Implementation of the abstract function defined in the base class. """
        with tf.GradientTape() as tape:
            prediction = self.model(X, training=True)
            # tf.print("prediction batch: ", prediction.shape)
            # tf.print("y batch: ", y.shape)
            loss = self.metrics_object.generic_loss()(y_true=y, y_pred=prediction, model_name=self.model_name,
                                                      epoch=epoch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Compute additional metrics

        if self.parameter.task_type == 'Classification':
            accuracy = self.metrics_object.accuracy(y_true=y, y_pred=prediction, model_name=self.model_name,
                                                    epoch=epoch)
            return [loss, accuracy]
        else:
            return [loss]

    # @tf.function
    def validate_on_batch(self, X, y, epoch):
        """ Implementation of the abstract function defined in the base class. """
        prediction = self.model(X, training=False)
        loss = self.metrics_object.generic_loss()(y_true=y, y_pred=prediction, model_name=self.model_name, epoch=epoch)

        # Compute additional metrics
        if self.parameter.task_type == 'Classification':
            accuracy = self.metrics_object.accuracy(y_true=y, y_pred=prediction, model_name=self.model_name,
                                                    epoch=epoch)
            return [loss, accuracy]
        else:
            return [loss]
