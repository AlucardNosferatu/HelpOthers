import tensorflow as tf


def loss_fn(y_true, y_pred):
    loss = tf.math.reduce_mean((y_true - y_pred) ** 2)
    return loss
