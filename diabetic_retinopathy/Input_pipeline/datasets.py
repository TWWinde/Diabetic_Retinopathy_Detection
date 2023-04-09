import gin
import logging
import tensorflow as tf
from keras import datasets as tfds


class DatasetInfo:
    input_shape = (32, 256, 256, 3)
    n_classes = 2
    fc_units = 32
    filters_num = 32
    dropout_rate = 0.3
    layer_dim = (1, 1, 1, 1)

    def __init__(self, input_shape, n_classes, fc_units, filters_num, dropout_rate, layer_dim):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.fc_units = fc_units
        self.filters_num = filters_num
        self.dropout_rate = dropout_rate
        self.layer_dim = layer_dim


DatasetInfo = DatasetInfo((256, 256, 3), 2, 32, 32, 0.3, (1, 1, 1, 1))


def read_labeled_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "label": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, features)
    image = tf.io.decode_jpeg(example['image'], channels=3) / 255
    label = example['label']

    return image, label  # returns a dataset of (image, label) pairs


def get_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTOTUNE)

    return dataset


@gin.configurable
def load(name, save_path):  # (name, data_dir)
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")
        ds_test = get_dataset(save_path + 'test.tfrecords')
        ds_train = get_dataset(save_path + 'train.tfrecords')
        ds_val = get_dataset(save_path + 'validation.tfrecords')

        ds_info = DatasetInfo

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tf.keras.datasets.mnist.load_data(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    if caching:
        ds_train = ds_train.cache()

    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
