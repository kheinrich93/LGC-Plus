import tensorflow as tf
import os

def specify_tf(use_tf,MEM_LIMIT):
    if use_tf:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gpus = tf.config.experimental.list_physical_devices('GPU')
        config = tf.config.experimental.set_memory_growth(gpus[0], True)
        MEMORY_LIMIT = MEM_LIMIT
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])
            except RuntimeError as e:
                print(e)
