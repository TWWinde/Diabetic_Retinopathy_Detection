import gin
import tensorflow as tf
from keras import layers
from keras import regularizers
from keras import Sequential


@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.
    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)
    Returns:
        (Tensor): output of the VGG block
    """
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = layers.BatchNormalization()(out)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out


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


@gin.configurable
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
