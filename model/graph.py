import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Conv2D, Conv2DTranspose, Flatten, Dense, Dropout, Activation
from tensorflow.keras.layers import MaxPooling3D, MaxPool2D, AveragePooling3D, Concatenate, Add, BatchNormalization


class CVANet:

    def get_model(self, parameter):
        # ---------------------------
        # Define network architecture
        # ---------------------------
        inputs = Input(shape=(None, None, parameter.cost_volume_depth, 1))
        inter_layer = inputs
        inter_layer = BatchNormalization()(inter_layer)

        # Neighbourhood layers
        nb_layer_num = int((parameter.nb_size - 1) / 2)
        for nb_layers in range(0, nb_layer_num):
            inter_layer = Conv3D(parameter.nb_filter_num, (3, 3, 3), kernel_initializer='random_normal')(inter_layer)
            inter_layer = BatchNormalization()(inter_layer)
            inter_layer = Activation('relu')(inter_layer)

        # Depth layers
        depth = 8
        for depth_layers in range(0, parameter.depth_layer_num):
            inter_layer = Conv3D(parameter.depth_filter_num, (1, 1, depth), padding='same',
                                 kernel_initializer='random_normal')(inter_layer)
            inter_layer = BatchNormalization()(inter_layer)
            inter_layer = Activation('relu')(inter_layer)

            if (depth < 64):
                depth = depth * 2

        # Dense layer - Fully convolutional
        dense_depth = parameter.cost_volume_depth - (2 * nb_layer_num)
        inter_layer = Conv3D(parameter.dense_filter_num, (1, 1, dense_depth), padding='valid',
                             kernel_initializer='glorot_normal', activation="relu")(inter_layer)
        inter_layer = Dropout(0.5)(inter_layer)

        for dense_layer in range(0, parameter.dense_layer_num):
            inter_layer = Conv3D(parameter.dense_filter_num, (1, 1, 1), padding='valid',
                                 kernel_initializer='glorot_normal', activation="relu")(inter_layer)
            inter_layer = Dropout(0.5)(inter_layer)

        inter_layer = Conv3D(1, (1, 1, 1), padding='valid', kernel_initializer='glorot_normal')(inter_layer)

        if parameter.task_type == 'Classification':
            predictions = Activation('sigmoid')(inter_layer)
        elif parameter.task_type == 'Regression':
            predictions = inter_layer
        else:
            raise Exception('Unknown task type: %s' % parameter.task_type)

        return Model(inputs=inputs, outputs=predictions)


class ConfNet:
    @staticmethod
    def encoding_unit(conv, filters, name, kernel_size=3, pool_size=(2, 2)):
        conv2d = Conv2D(filters, kernel_size, padding="SAME", name=name)(conv)
        conv2dBN = BatchNormalization()(conv2d)
        conv2d_relu = tf.nn.relu(conv2dBN)

        conv2d_maxPool = MaxPool2D(pool_size=pool_size)(conv2d_relu)
        forward = conv2d

        return conv2d_maxPool, forward

    @staticmethod
    def decoding_unit(conv, filters, name, use_BN, kernel_size=(3, 3), forwards=None, stride=2):
        deconv2d = Conv2DTranspose(filters, kernel_size, strides=(stride, stride), padding="SAME", activation=None,
                                   name=name + "_DECON")(conv)
        if forwards is not None:
            if isinstance(forwards, (list, tuple)):
                for f in forwards:
                    deconv2d = Concatenate()([deconv2d, f])
            else:
                deconv2d = Concatenate()([deconv2d, forwards])

        if use_BN:
            conv2dBN = BatchNormalization()(deconv2d)
            conv2d = Conv2D(filters, kernel_size, padding="SAME", name=name + "_CONV")(conv2dBN)
        else:
            conv2d = Conv2D(filters, kernel_size, padding="SAME", name=name + "_CONV")(deconv2d)

        conv2d_relu = tf.nn.relu(conv2d)

        return conv2d_relu

    def get_model(self, parameter):
        # ---------------------------
        # Define network architecture
        # ---------------------------
        inputs_disp = Input(shape=(None, None, 1), name='INPUT_DISP')
        if parameter.use_warp == 'EF':
            inputs_rgb = Input(shape=(None, None, 4), name='INPUT_RGB')
        elif parameter.use_warp == 'LF':
            inputs_rgb = Input(shape=(None, None, 3), name='INPUT_RGB')
            inputs_warp = Input(shape=(None, None, 1), name='INPUT_WARP')
        else:
            inputs_rgb = Input(shape=(None, None, 3), name='INPUT_RGB')

        filters = 32
        kernel_size = 3

        # Encoding
        conv1_RGB = Conv2D(filters, kernel_size, strides=(1, 1), padding='same', activation="relu", name='CONV1_RGB')(
            inputs_rgb)
        x = Model(inputs=inputs_rgb, outputs=conv1_RGB)

        conv1_disp = Conv2D(filters, kernel_size, activation="relu", strides=(1, 1), padding='same', name='CONV1_DISP')(
            inputs_disp)
        y = Model(inputs=inputs_disp, outputs=conv1_disp)

        if parameter.use_warp == 'LF':
            conv1_warp = Conv2D(filters, kernel_size, activation="relu", strides=(1, 1), padding='same',
                                name='CONV1_WARP')(
                inputs_warp)
            z = Model(inputs=inputs_warp, outputs=conv1_warp)

        if parameter.use_warp == 'LF':
            concat = Concatenate()([x.output, y.output, z.output])
        else:
            concat = Concatenate()([x.output, y.output])

        # Encoding block,4x: 3x3 conv2d, 2x2 max pooling each, stride 1, RELU
        encoding1, scale1 = ConfNet.encoding_unit(concat, filters * 2, "ENCODING1")
        encoding2, scale2 = ConfNet.encoding_unit(encoding1, filters * 4, "ENCODING2")
        encoding3, scale3 = ConfNet.encoding_unit(encoding2, filters * 8, "ENCODING3")
        encoding4, scale4 = ConfNet.encoding_unit(encoding3, filters * 16, "ENCODING4")

        # Decoding
        # Decoding block,4x: 3x3 decon 2d, 3x3 conv2d, stride 2, RELU
        decoding5 = ConfNet.decoding_unit(encoding4, filters * 8, "DECODING1", parameter.use_BN, forwards=scale4)
        decoding6 = ConfNet.decoding_unit(decoding5, filters * 4, "DECODING2", parameter.use_BN, forwards=scale3)
        decoding7 = ConfNet.decoding_unit(decoding6, filters * 2, "DECODING3", parameter.use_BN, forwards=scale2)
        decoding8 = ConfNet.decoding_unit(decoding7, filters, "DECODING4", parameter.use_BN, forwards=concat)

        # Sigmoid classification
        predictions = Conv2D(1, kernel_size, padding='same', activation='sigmoid')(decoding8)

        if parameter.use_warp == 'LF':
            return Model(inputs=[x.input, y.input, z.input], outputs=predictions)
        else:
            return Model(inputs=[x.input, y.input], outputs=predictions)


class LGC:
    @staticmethod
    def conv_tower(x, filters, kernel_size, conv_num, name, CNN_mode, is_disp=True, scale=1):
        if is_disp:
            conv = x
        else:
            conv = x * scale

        # Padding: valid for training/same for testing
        if CNN_mode == 'Training':
            padding = 'valid'
        else:
            padding = 'same'

        for i in range(0, conv_num):
            conv = Conv2D(filters, kernel_size, activation="relu", padding=padding, name=name + str(i + 1))(conv)

        return conv

    def get_model(self, parameter):
        # ---------------------------
        # Define network architecture
        # ---------------------------
        inputs_disp = Input(shape=(None, None, 1), name='INPUT_DISP')
        inputs_local = Input(shape=(None, None, 1), name='INPUT_LOCAL')
        inputs_global = Input(shape=(None, None, 1), name='INPUT_GLOBAL')

        filters = 64
        kernel_size = 3
        fc_filters = 100
        scale = 1.0
        conv_num = 4

        # Combining three input tower for disp, local global: 4x 3x3 conv2D, relu
        conv_disp = LGC.conv_tower(inputs_disp, filters, kernel_size, conv_num, "CONV_DISP_",
                                   CNN_mode=parameter.CNN_mode)
        x = Model(inputs=inputs_disp, outputs=conv_disp)

        conv_local = LGC.conv_tower(inputs_local, filters, kernel_size, conv_num, "CONV_LOCAL_",
                                    CNN_mode=parameter.CNN_mode, is_disp=False, scale=scale)
        y = Model(inputs=inputs_local, outputs=conv_local)

        conv_global = LGC.conv_tower(inputs_global, filters, kernel_size, conv_num, "CONV_GLOBAL_",
                                     CNN_mode=parameter.CNN_mode, is_disp=False, scale=scale)
        z = Model(inputs=inputs_global, outputs=conv_global)

        concat = Concatenate()([x.output, y.output, z.output])

        # Two fully connected layer kernel
        fc1 = Conv2D(fc_filters, 1, activation="relu", padding='valid', name='FC1')(concat)
        fc2 = Conv2D(fc_filters, 1, activation="relu", padding='valid', name='FC2')(fc1)

        # Sigmoid classification
        prediction = Conv2D(1, 1, activation='sigmoid', padding='valid', name='PREDICTION')(fc2)

        return Model(inputs=[x.input, y.input, z.input], outputs=prediction)


class LFN:
    @staticmethod
    def conv_tower(conv, filters, kernel_size, conv_num, name, CNN_mode):

        if CNN_mode == 'Training':
            padding = 'valid'
        else:
            padding = 'same'

        for i in range(0, conv_num):
            conv = Conv2D(filters, kernel_size, activation="relu", padding='valid', name=name + str(i + 1))(conv)

        return conv

    def get_model(self, parameter):
        # ---------------------------
        # Define network architecture
        # ---------------------------
        inputs_disp = Input(shape=(None, None, 1), name='INPUT_DISP')
        inputs_rgb = Input(shape=(None, None, 3), name='INPUT_RGB')

        filters = 64
        kernel_size = 3
        fc_filters = 100
        conv_num = 4

        # Combining two input tower
        conv_disp = LFN.conv_tower(inputs_disp, filters, kernel_size, conv_num, "CONV_DISP_", parameter.CNN_mode)
        x = Model(inputs=inputs_disp, outputs=conv_disp)

        conv_rgb = LFN.conv_tower(inputs_rgb, filters, kernel_size, conv_num, "CONV_RGB_", parameter.CNN_mode)
        y = Model(inputs=inputs_rgb, outputs=conv_rgb)

        concat = Concatenate()([x.output, y.output])

        # Two fully connected layer kernel
        fc1 = Conv2D(fc_filters, 1, activation="relu", padding='valid', name='FC1')(concat)
        fc2 = Conv2D(fc_filters, 1, activation="relu", padding='valid', name='FC2')(fc1)

        # Sigmoid classification
        prediction = Conv2D(1, 1, activation='sigmoid', padding='valid', name='PREDICTION')(fc2)

        return Model(inputs=[x.input, y.input], outputs=prediction)
