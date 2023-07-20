import os
import random

import numpy as np
import tensorflow as tf

from Model.BertDNN.Bert import build_processor
from Model.BertDNN.TopicsClassifier import data_load_topics
from Model.GetSentiment import data_load_sentiments


def pad_to_length(topics, sentiments, pad_length):
    topics = topics.tolist()
    sentiments = sentiments.tolist()
    while len(topics) < pad_length:
        assert len(topics) == len(sentiments)
        topics.append([0.0] * 7)
        sentiments.append(0.0)
    topics = np.array(topics)
    sentiments = np.array(sentiments)
    return topics, sentiments


def data_load_single_seq(data_path='../../Data/chat.txt', pad_length=32, load_amount=32):
    def process_per_index(i, t, s, pl):
        y1_, y2_ = np.copy(t[i, :]), np.copy(s[i])
        if pl is None:
            x1_, x2_ = np.copy(t[:i, :]), np.copy(s[:i])
        else:
            x1_, x2_ = pad_to_length(np.copy(t[:i, :]), np.copy(s[:i]), pl)
        return y1_, y2_, x1_, x2_

    processor = build_processor(use_post_trained=False)
    _, topics = data_load_topics(processor=processor, data_path=data_path)
    sentiments = data_load_sentiments(data_path=data_path)
    load_indices = []
    if load_amount is None:
        y_t, y_s, x_t, x_s = None, None, topics, sentiments
    else:
        while len(load_indices) < load_amount:
            load_indices.append(random.choice(list(range(topics.shape[0]))))
        y_t = []
        y_s = []
        x_t = []
        x_s = []
        for index in load_indices:
            y1, y2, x1, x2 = process_per_index(index, topics, sentiments, pad_length)
            y_t.append(y1)
            y_s.append(y2)
            x_t.append(x1)
            x_s.append(x2)
    return y_t, y_s, x_t, x_s


def data_generate(y_t, y_s, x_t, x_s, batch_size=32):
    while True:
        x_t_list = []
        x_s_list = []
        y_t_list = []
        y_s_list = []
        while len(x_t_list) < batch_size:
            assert len(x_t_list) == len(x_s_list) == len(y_t_list) == len(y_s_list)
            index = random.choice(list(range(len(y_t))))
            x_t_list.append(x_t[index])
            x_s_list.append(x_s[index])
            y_t_list.append(y_t[index])
            y_s_list.append(y_s[index])
        yield [np.array(x_t_list), np.array(x_s_list)], [np.array(y_t_list), np.array(y_s_list)]


def model_train(model, batch_size=32):
    y_t, y_s, x_t, x_s = data_load_single_seq(data_path='../../Data/chat.txt', pad_length=32, load_amount=32)
    data_gen = data_generate(y_t, y_s, x_t, x_s, batch_size=batch_size)
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
        model.fit(
            x=data_gen,
            epochs=10,
            callbacks=[ckpt],
            steps_per_epoch=100
        )


def model_test(model):
    y_t, y_s, x_t, x_s = data_load_single_seq(data_path='../../Data/chat.txt', pad_length=32, load_amount=32)
    data_gen = data_generate(y_t, y_s, x_t, x_s, batch_size=1)
    flag = True
    while flag:
        x, y = data_gen.__next__()
        res = model.predict(x=x)
        print('pred topic:', ['%.3f' % item for item in res[0][0, :].tolist()])
        print('true topic:', ['%.3f' % item for item in y[0][0, :].tolist()])
        print('diff topic:', ['%.3f' % abs(item) for item in (res[0][0, :] - y[0][0, :]).tolist()])
        print(
            'diff topic average:',
            '%.3f' % np.average(
                np.array(
                    [abs(item) for item in (res[0][0, :] - y[0][0, :]).tolist()]
                )
            )
        )
        print('pred sentiment:', '%.3f' % res[1][0])
        print('true sentiment:', '%.3f' % y[1][0])
        print('diff sentiment:', '%.3f' % abs(res[1][0] - y[1][0]))
        cmd = ''
        while cmd not in ['y', 'n']:
            cmd = input('continue?:y/n\n')
        flag = (cmd == 'y')


def model_build(force_new=False, model_path='NextPredictor.h5'):
    if not os.path.exists(model_path) or force_new:
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
        output_sentiments_seq = tf.keras.layers.Dense(1)(x)
        output_sentiments_seq = tf.keras.backend.squeeze(output_sentiments_seq, axis=-1)
        output_sentiments_seq = tf.keras.layers.Activation(activation="sigmoid", name='output_sentiments')(
            output_sentiments_seq
        )
        model = tf.keras.Model([input_topics_seq, input_sentiments_seq], [output_topics_seq, output_sentiments_seq])
    else:
        print('Use old model')
        model = tf.keras.models.load_model(model_path)
    return model


if __name__ == '__main__':
    mdl = model_build(force_new=False)
    # model_train(mdl, batch_size=8)
    model_test(mdl)
