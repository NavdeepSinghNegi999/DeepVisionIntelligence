# src/models/caption_model.py

import tensorflow as tf
from tensorflow import keras


class ImageCaptioningModel(keras.Model):
    """
    Custom Keras model for image captioning with
    CNN encoder + Transformer encoder/decoder.
    """

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")

    def call(self, inputs):
        images, training, captions = inputs
        features = self.cnn_model(images)
        encoded = self.encoder(features, training=False)
        return self.decoder(captions, encoded, training=training, mask=None)

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)  
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    def calculate_accuracy(self, y_true, y_pred, mask):
        acc = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        acc = tf.logical_and(mask, acc)
        acc = tf.cast(acc, tf.float32)
        return tf.reduce_sum(acc) / tf.reduce_sum(tf.cast(mask, tf.float32))

    def _compute_loss_and_acc(self, image_features, captions, training):
        encoded = self.encoder(image_features, training=training)

        seq_inp = captions[:, :-1]
        seq_true = captions[:, 1:]
        mask = tf.math.not_equal(seq_true, 0)

        preds = self.decoder(seq_inp, encoded, training=training, mask=mask)

        loss = self.calculate_loss(seq_true, preds, mask)
        acc = self.calculate_accuracy(seq_true, preds, mask)

        return loss, acc

    def train_step(self, data):
        images, captions = data

        if self.image_aug:
            images = self.image_aug(images)

        image_features = self.cnn_model(images)

        with tf.GradientTape() as tape:
            loss, acc = self._compute_loss_and_acc(image_features, captions, True)

        train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    def test_step(self, data):
        images, captions = data
        image_features = self.cnn_model(images)

        loss, acc = self._compute_loss_and_acc(image_features, captions, False)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
