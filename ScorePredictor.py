import os.path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from DataReader import read_file, unify_symbol, extract_parenthesis
from TextEncoder import build_processor, tokenize


def data_load(new_data=False):
    if new_data:
        all_input, all_output = read_file(tokenize_batch=64)
        all_input = np.array(all_input)
        all_output = np.array(all_output)
        np.save('Input.npy', all_input)
        np.save('Output.npy', all_output)
    else:
        all_input = np.load('Input.npy')
        all_output = np.load('Output.npy')
    return all_input, all_output


def model_build():
    inputs = tf.keras.Input(shape=(32,))
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(inputs)
    x = tf.keras.layers.Dense(512, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.tanh)(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.tanh)(x)
    outputs = tf.keras.layers.Dense(5)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def model_train(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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
        model.fit(x=all_input_, y=all_output_, batch_size=512, epochs=10000, callbacks=[ckpt])


def model_test(model, processor=None, data_file_path='my_personality.csv', least_words=3, most_word=30):
    if processor is None:
        processor = build_processor(seq_len=32)
    data_csv = pd.read_csv(data_file_path)
    flag = True
    while flag:
        index = random.choice(list(range(data_csv.shape[0])))
        row = data_csv.iloc[index, :]
        text = row['STATUS']
        s_ext = row['sEXT']
        s_neu = row['sNEU']
        s_agr = row['sAGR']
        s_con = row['sCON']
        s_opn = row['sOPN']
        score = [s_ext, s_neu, s_agr, s_con, s_opn]
        text = unify_symbol(text)
        texts = extract_parenthesis(text)
        for text in texts:
            text_slices = text.split('.')
            for text_slice in text_slices:
                if least_words < len(text_slice.split(' ')) < most_word:
                    print('text:', text_slice.lower())
                    embedding = tokenize([text_slice.lower()], processor)
                    res = model.predict(x=embedding)
                    print('pred:', res.tolist()[0])
                    print('true:', score)
                    print('diff:', np.array(score) - res[0, :])
        cmd = ''
        while cmd not in ['y', 'n']:
            cmd = input('continue?:y/n\n')
        flag = (cmd == 'y')


if __name__ == '__main__':
    train = True
    test = not train
    all_input_, all_output_ = data_load()
    if os.path.exists('ScorePredictor.h5'):
        model_ = tf.keras.models.load_model('ScorePredictor.h5')
    else:
        model_ = model_build()
    if train:
        model_train(model_)
    if test:
        model_test(model_)
