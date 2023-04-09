import gin
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras import regularizers
from keras import Sequential


@gin.configurable
def res_basic_block(inputs, filters, kernel_size, strides):  # filters = num(kernels)
    out = layers.Conv2D(filters, kernel_size, padding="same", strides=strides,
                        kernel_regularizer=regularizers.l1(0.01))(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    out = layers.Conv2D(filters, kernel_size, padding="same", strides=1,
                        kernel_regularizer=regularizers.l1(0.01))(out)
    out = layers.BatchNormalization()(out)

    if strides != 1:
        identity = layers.Conv2D(filters, kernel_size=1,
                                 padding="same", strides=strides)(inputs)
    else:
        identity = layers.Conv2D(filters, kernel_size=1,
                                 padding="same", strides=1)(inputs)

    out = layers.add([out, identity])  # add identity if stride=2 it has similar effect with pooling downsampling
    out = layers.ReLU()(out)
    return out


@gin.configurable  # block_num：build里面包含basic的个数  base_filters: kernel的个数
def res_build_block(inputs, base_filters, block_num, strides, dropout_rate):
    out = res_basic_block(inputs, filters=base_filters, strides=strides)
    for i in range(1, block_num):
        out = res_basic_block(out, filters=base_filters, strides=1)
    out = layers.Dropout(dropout_rate)(out)
    return out


@gin.configurable
def res_stem(input_shape):
    return Sequential(
        [
            tf.keras.Input(shape=input_shape),
            layers.Conv2D(64, 8, 2, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same"),
        ], name="stem"
    )
