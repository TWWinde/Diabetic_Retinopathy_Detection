import gin
import tensorflow as tf
from keras import layers
from keras import regularizers
from models_Resnet.layers import res_stem, res_build_block, res_basic_block


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
