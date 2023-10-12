import gin
import os
import tensorflow as tf
from numpy import *
from input_pipeline.data_prepare import get_image_names_labels
import numpy as np


def get_images(data_dir, image_name):
    image = open(data_dir + image_name + '.jpg', 'rb').read()
    return image


# Use this function to serialize images and labels into the TFRecord format
@gin.configurable
def write_Tfrecord(save_path):
    test_img_path = os.path.join(save_path, 'image', 'test')
    test_label_imagename = get_image_names_labels(save_path + 'test.csv')
    with tf.io.TFRecordWriter(save_path + 'test.tfrecords') as writer:
        for i in range((len(test_label_imagename))):
            image_raw = get_images(test_img_path, test_label_imagename[i, 0])
            feature = {  # build Feature dictionary
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[test_label_imagename[i, 1]]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    train_img_path = save_path + 'image/train/'
    train_label_imagename = get_image_names_labels(save_path + 'train.csv')
    train_label_imagename = np.random.permutation(train_label_imagename)
    with tf.io.TFRecordWriter(save_path + 'train.tfrecords') as writer:
        for i in range(int((len(train_label_imagename) * 0.8))):
            image_raw = get_images(train_img_path, train_label_imagename[i, 0])
            feature = {  # build Feature dictionary
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[train_label_imagename[i, 1]]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    with tf.io.TFRecordWriter(save_path + 'validation.tfrecords') as writer:
        for i in range(int(0.8 * len(train_label_imagename)), len(train_label_imagename)):
            image_raw = get_images(train_img_path, train_label_imagename[i, 0])
            feature = {  # build Feature dictionary
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[train_label_imagename[i, 1]]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

