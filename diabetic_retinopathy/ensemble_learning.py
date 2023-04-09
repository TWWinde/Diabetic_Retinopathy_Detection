import tensorflow as tf
import gin
import logging
from absl import app
from utilss import utils_params, utils_misc
from input_pipeline.datasets import load
from models.architectures import vgg_like
from models_Resnet.architectures import resnet
from models.TL import tl_inception, tl_xception, tl_inception_resnet


def ensemble(model_list, ds_val, ds_test):
    logging.info(f'======== Starting Ensemble ========')

    for name, ds in [('val', ds_val), ('test', ds_test)]:
        for images, labels in ds:

            predictions = [model(images, training=False) for model in model_list]
            predictions = tf.squeeze(tf.reduce_mean(tf.convert_to_tensor(predictions), axis=0))
            print(predictions)
            eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
            eval_accuracy(labels, predictions)

        template = ' Validation Accuracy: {}'
        logging.info(template.format(eval_accuracy.result() * 100))

    logging.info('======== Finished Ensemble Evaluation ========')


def main(argv):
    run_paths = utils_params.gen_run_folder("ensemble")
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.DEBUG)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    # utils_params.save_config(run_paths['path_gin'], gin.config_str())

    ds_train, ds_val, ds_test, ds_info = load()

    vgg_trained = vgg_like()
    checkpoint_1 = tf.train.Checkpoint(step=tf.Variable(0), model=vgg_trained)
    manager = tf.train.CheckpointManager(checkpoint_1,
                                         directory="/Users/mengxiangyuan/Desktop/Deep_Learning_Lab/WS2022/dl-lab-22w"
                                                   "-team08-master/Results/01_VGG/run_2022-12-13T13-03-50-977119"
                                                   "/ckpts/",
                                         max_to_keep=3)
    checkpoint_1.restore(manager.latest_checkpoint)

    resnet_trained = resnet(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    checkpoint_2 = tf.train.Checkpoint(step=tf.Variable(0), model=resnet_trained)
    manager = tf.train.CheckpointManager(checkpoint_2,
                                         directory="//Users/mengxiangyuan/Desktop/Deep_Learning_Lab/WS2022/dl-lab-22w"
                                                   "-team08-master/Results/02_Resnet/ckpt_resnet/ckpts",
                                         max_to_keep=3)
    checkpoint_2.restore(manager.latest_checkpoint)

    tl_inception_trained = tl_inception(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    checkpoint_3 = tf.train.Checkpoint(step=tf.Variable(0), model=tl_inception_trained)
    manager = tf.train.CheckpointManager(checkpoint_3,
                                         directory="/Users/mengxiangyuan/Desktop/Deep_Learning_Lab/WS2022/dl-lab-22w"
                                                   "-team08-master/Results/03_tl_inception/ckpt_tl_inception/ckpts",
                                         max_to_keep=3)
    checkpoint_3.restore(manager.latest_checkpoint)

    tl_xception_trained = tl_xception(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    checkpoint_4 = tf.train.Checkpoint(step=tf.Variable(0), model=tl_xception_trained)
    manager = tf.train.CheckpointManager(checkpoint_4,
                                         directory="",
                                         max_to_keep=3)
    checkpoint_4.restore(manager.latest_checkpoint)

    tl_inception_resnet_trained = tl_inception_resnet(input_shape=ds_info.input_shape, n_classes=ds_info.n_classes)
    checkpoint_5 = tf.train.Checkpoint(step=tf.Variable(0), model=tl_inception_resnet_trained)
    manager = tf.train.CheckpointManager(checkpoint_5,
                                         directory="",
                                         max_to_keep=3)
    checkpoint_5.restore(manager.latest_checkpoint)

    model_list = [vgg_trained, resnet_trained, tl_inception_trained, tl_xception_trained, tl_inception_resnet_trained]
    ensemble(model_list, ds_val, ds_test)


if __name__ == "__main__":
    app.run(main)
