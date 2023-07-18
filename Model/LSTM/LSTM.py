import random

import numpy as np
import tensorflow as tf

from Model.BertDNN.Bert import build_processor
from Model.BertDNN.TopicsClassifier import data_load_topics
from Model.GetSentiment import data_load_sentiments


def data_load_single_seq(data_path='../../Data/chat.txt', pad_length=32, batch_size=32):
    def pad_to_length(t, s, pl):
        t = t.tolist()
        s = s.tolist()
        while len(t) < pl:
            assert len(t) == len(s)
            t.append([0.0] * 7)
            s.append(0.0)
        t = np.array(t)
        s = np.array(s)
        return t, s

    def process_per_step(i, t, s, pl):
        y1_, y2_ = np.copy(t[i, :]), np.copy(s[i])
        x1_, x2_ = pad_to_length(np.copy(t[:i, :]), np.copy(s[:i]), pl)
        return y1_, y2_, x1_, x2_

    processor = build_processor(use_post_trained=False)
    _, topics = data_load_topics(processor=processor, data_path=data_path)
    sentiments = data_load_sentiments(data_path=data_path)
    batch_indices = []
    while len(batch_indices) < batch_size:
        batch_indices.append(random.choice(list(range(topics.shape[0]))))
    y_t = []
    y_s = []
    x_t = []
    x_s = []
    for index in batch_indices:
        y1, y2, x1, x2 = process_per_step(index, topics, sentiments, pad_length)
        y_t.append(y1)
        y_s.append(y2)
        x_t.append(x1)
        x_s.append(x2)
    return y_t, y_s, x_t, x_s


def model_train(model, y_t, y_s, x_t, x_s):
    def gen_batch_x(y_t_, y_s_, x_t_, x_s_):
        while True:
            index = random.choice(list(range(len(xt))))
            yield [
                np.expand_dims(x_t_[index], axis=0),
                np.expand_dims(x_s_[index], axis=0)], [
                np.expand_dims(y_t_[index], axis=0),
                np.expand_dims(y_s_[index], axis=0)
            ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss={
            'output_topics': tf.keras.losses.categorical_crossentropy,
            'output_sentiments': tf.keras.losses.mean_squared_error
        },
        metrics=['accuracy'],
        run_eagerly=True
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='NextPredictor.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
    )
    with tf.device('/gpu:0'):
        model.fit(x=gen_batch_x(y_t, y_s, x_t, x_s), batch_size=2048, epochs=10, callbacks=[ckpt], steps_per_epoch=100)


def model_build():
    input_topics_seq = tf.keras.Input(shape=(32, 7))
    input_sentiments_seq = tf.keras.Input(shape=(32,))
    # 32*7->32*128
    x1 = tf.keras.layers.Dense(128)(input_topics_seq)
    # 32->32*128
    x2 = tf.keras.layers.Embedding(32, 128)(input_sentiments_seq)
    # 32*128+32*128->32*256
    x = tf.keras.backend.concatenate([x1, x2], axis=-1)
    # Add 2 bidirectional LSTMs
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    # Add a classifier
    output_topics_seq = tf.keras.layers.Dense(7, activation="softmax", name='output_topics')(x)
    output_sentiments_seq = tf.keras.layers.Dense(1, activation="sigmoid", name='output_sentiments')(x)
    model = tf.keras.Model([input_topics_seq, input_sentiments_seq], [output_topics_seq, output_sentiments_seq])
    return model


if __name__ == '__main__':
    mdl = model_build()
    yt, ys, xt, xs = data_load_single_seq()
    model_train(mdl, yt, ys, xt, xs)
