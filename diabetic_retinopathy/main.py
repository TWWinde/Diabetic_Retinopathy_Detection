import os
import random
import gin
import logging
import numpy as np
from absl import app, flags
import tensorflow as tf
from Deep_visualization.Dimensionality_reduction import Dimensionality_Reduction
from Input_pipeline.TFrecord_writer import write_Tfrecord
from Input_pipeline.data_prepare import processing_augmentation_oversampling
from train import Trainer
from evaluation.eval import evaluate, evaluate_fl
from Input_pipeline.datasets import load, get_dataset
from utilss import utils_params, utils_misc
from models.architectures import vgg_like
from models_Resnet.architectures import resnet
from models.TL import tl_inception, tl_xception, tl_inception_resnet
from evaluation.metrics import confusionmatrix, ROC


# fix the seed to make the training repeatable
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


setup_seed(66)

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or test a model.')  # train


FLAG1 = flags.FLAGS
flags.DEFINE_boolean('TFrecord_needed', False, 'Specify if TFRecord is ready.')


@gin.configurable
def main(argv):
    # change the number to decide which model.
    Choose_model = ['vgg_like', 'resnet', 'tl_inception', 'tl_xception', 'tl_inception_resnet']
    model_flag = Choose_model[0]

    # change the number to decide the operation.
    Choose = ['evaluate_fl', 'confusionmatrix', 'Dimensionality_Reduction', 'ROC']
    test_flag = Choose[0]

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    if FLAG1.TFrecord_needed:
        # get images and labels
        processing_augmentation_oversampling()
        # write TFRecord
        write_Tfrecord()
    else:
        pass

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    if model_flag == 'vgg_like':
        model = vgg_like()
    elif model_flag == 'resnet':
        model = resnet(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    elif model_flag == 'tl_inception':
        model = tl_inception(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    elif model_flag == 'tl_xception':
        model = tl_xception(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    elif model_flag == 'tl_inception_resnet':
        model = tl_inception_resnet(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)

    model.summary()

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    else:
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
        manager = tf.train.CheckpointManager(checkpoint,
                                             directory="/misc/home/RUS_CIP/st180408/dl-lab-22w-team08/experiments/run_2023-02-10T04-06-41-658852/ckpts/",
                                             max_to_keep=3)


        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            tf.print("restore")
        else:
            tf.print("bad")

        if test_flag == 'evaluate_fl':
            ds_test = ds_test.batch(32)
            evaluate_fl(model, ds_test)
        elif test_flag == 'confusionmatrix':
            confusionmatrix(model, ds_test)
        elif test_flag == 'Dimensionality_Reduction':
            Dimensionality_Reduction(model, ds_test)
        elif test_flag == 'ROC':
            ROC(model, ds_test)


if __name__ == "__main__":
    app.run(main)
