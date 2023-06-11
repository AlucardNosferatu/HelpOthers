import os.path

import numpy as np
import tensorflow as tf

from DataReader import read_file

if __name__ == '__main__':
    new_data = False
    if new_data:
        all_input, all_output = read_file(tokenize_batch=64)
        all_input = np.array(all_input)
        all_output = np.array(all_output)
        np.save('Input.npy', all_input)
        np.save('Output.npy', all_output)
    else:
        all_input = np.load('Input.npy')
        all_output = np.load('Output.npy')
    if os.path.exists('ScorePredictor.h5'):
        model = tf.keras.models.load_model('ScorePredictor.h5')
    else:
        inputs = tf.keras.Input(shape=(32,))
        x = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(16, activation=tf.nn.leaky_relu)(x)
        outputs = tf.keras.layers.Dense(5, activation=tf.nn.relu)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='ScorePredictor.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
    )
    with tf.device('/gpu:0'):
        model.fit(x=all_input, y=all_output, batch_size=256, epochs=10000, callbacks=[ckpt])
