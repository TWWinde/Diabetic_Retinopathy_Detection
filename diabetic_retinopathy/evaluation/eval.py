import gin
import numpy as np
import tensorflow as tf
import logging


# This function is for the evaluation of model with SparseCategoricalCrossentropy
def evaluate(model, ds_test):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
    eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    for images, labels in ds_test:
        print('groundtruth', labels.numpy())
        predictions = model(images, training=False)
        print('prediction.', np.argmax(predictions.numpy(), axis=-1))
        print('..............................................................')
        t_loss = loss_object(labels, predictions)
        eval_loss(t_loss)
        eval_accuracy(labels, predictions)

    template = ' Test Loss: {}, Test Accuracy: {}'
    logging.info(template.format(eval_loss.result(), eval_accuracy.result() * 100))

    return


# This function is for the evaluation of model with Focal loss
def evaluate_fl(model, ds_test):
    def greater_than_0_5(x):
        return tf.where(x > 0.5, 1, 0)

    alpha = 0.25
    gamma = 1.0
    loss_object = tf.keras.losses.BinaryFocalCrossentropy(from_logits=False, alpha=alpha, gamma=gamma)
    eval_accuracy = tf.keras.metrics.BinaryAccuracy(name='eval_accuracy')
    eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    for images, labels in ds_test:
        print('groundtruth', labels.numpy())
        predictions = model(images, training=False)
        # print(predictions)
        results = greater_than_0_5(predictions)
        results = results.numpy()
        results = np.squeeze(results)
        print('prediction.', results)
        print('..............................................................')
        t_loss = loss_object(labels, predictions)
        eval_loss(t_loss)
        eval_accuracy(labels, predictions)

    template = ' Test Loss: {}, Test Accuracy: {}'
    logging.info(template.format(eval_loss.result(), eval_accuracy.result() * 100))

    return

