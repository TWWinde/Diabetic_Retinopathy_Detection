import datetime
import os

import gin
import numpy as np
import tensorflow as tf
import logging
from tensorflow import train
import tensorflow_addons as tfa


@gin.configurable
class Trainer(object):

    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval, acc,
                 alpha, gamma):
        # Summary Writer

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_dir = os.path.dirname(__file__)
        tensorboard_log_dir = os.path.join(current_dir, 'logs')
        log_dir = os.path.join(tensorboard_log_dir, current_time)
        logging.info(f"Tensorboard output will be stored in: {log_dir}")
        self.train_log_dir = os.path.join(log_dir, 'train')
        self.val_log_dir = os.path.join(log_dir, 'validation')

        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.alpha = alpha
        self.gamma = gamma
        # Loss objective
        # self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.loss_object = tf.keras.losses.BinaryFocalCrossentropy(from_logits=False, alpha=alpha, gamma=gamma)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001)
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001,
                                                                 decay_steps=1000,
                                                                 alpha=0.1)
        self.optimizer = tf.keras.optimizers.Adam(lr_scheduler)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        # self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.acc = acc
        # ....
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=self.model, optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=run_paths["path_ckpts_train"],
                                                  max_to_keep=3)
        # Checkpoint Manager
        # ...

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            # labels = tf.one_hot(indices=labels, depth=2, dtype=tf.float32)
            labels = tf.cast(labels, dtype=tf.float32)
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        # labels = tf.one_hot(indices=labels, depth=2, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.float32)
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def write_scalar_summary(self, step):
        """ Write scalar summary to tensorboard """

        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=step)
            tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)

        with self.val_summary_writer.as_default():
            tf.summary.scalar('loss', self.val_loss.result(), step=step)
            tf.summary.scalar('accuracy', self.val_accuracy.result(), step=step)

    def train(self):
        logging.info(self.model.summary())
        logging.info('\n================ Starting Training ================')
        self.acc = 0

        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)
            # logging.info('\nThe {} step is now being implemented.'.format(step))

            if step % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # Write summary to tensorboard
                self.write_scalar_summary(step)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                if self.acc < self.val_accuracy.result():
                    self.acc = self.val_accuracy.result()
                    logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                    path = self.manager.save()
                    print("model saved to %s" % path)
                    # Save checkpoint
                    # ...

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                path = self.manager.save()
                print("final model saved to %s" % path)
                # Save final checkpoint
                # ...
                return self.val_accuracy.result().numpy()

        template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
        logging.info(template.format(step,
                                     self.train_loss.result(),
                                     self.train_accuracy.result() * 100,
                                     self.val_loss.result(),
                                     self.val_accuracy.result() * 100))

        logging.info('\n================ Finished Training ================')


class Example:
    pass
