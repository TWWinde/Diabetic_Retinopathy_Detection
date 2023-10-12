import gin
import tensorflow as tf
from keras import layers
from models.layers import vgg_block
from keras import regularizers
from layers import res_stem, res_build_block, res_basic_block

@gin.configurable
def Basic_CNN(input_shape, base_filters, kernel_size, dense_units, dropout_rate, n_classes):
    """Defines a basic CNN Network as benchmark.
      in oder to validate the whole training precess
        """
    inputs = tf.keras.Input(input_shape)
    out = tf.keras.layers.Conv2D(base_filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Conv2D(base_filters * 2, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Conv2D(base_filters * 4, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.Conv2D(base_filters * 8, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='Basic_CNN')


@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate
    Returns:
        (keras.Model): keras model object
    """
    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')






@gin.configurable
def resnet(input_shape, n_classes, dense_units, dropout_rate):
    inputs = tf.keras.Input(shape=input_shape),
    out = res_stem(input_shape)(inputs)
    out = res_basic_block(out, 64, 3, strides=1)
    out = res_basic_block(out, 64, 3, strides=1)
    out = res_basic_block(out, 128, 3, strides=2)
    out = res_basic_block(out, 128, 3, strides=1)
    out = res_basic_block(out, 256, 3, strides=2)
    out = res_basic_block(out, 256, 3, strides=1)
    out = res_basic_block(out, 512, 3, strides=2)
    out = res_basic_block(out, 512, 3, strides=1)

    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)
    out = layers.Dropout(dropout_rate)(out)
    outputs = layers.Dense(n_classes, activation=tf.nn.softmax)(out)
    return tf.keras.Model(inputs, outputs, name="resnet")
