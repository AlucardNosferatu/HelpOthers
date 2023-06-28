import os.path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from Data.BertDNN.DataReader import read_file, unify_symbol, extract_parenthesis
from Bert import build_processor, embed


def data_load(new_data=False):
    if new_data:
        all_input, all_output = read_file(tokenize_batch=64)
        all_input = np.array(all_input)
        all_output = np.array(all_output)
        np.save('../../Data/BertDNN/Input.npy', all_input)
        np.save('../../Data/BertDNN/Output.npy', all_output)
    else:
        all_input = np.load('../../Data/BertDNN/Input.npy')
        all_output = np.load('../../Data/BertDNN/Output.npy')
    return all_input, all_output


def model_build():
    inputs = tf.keras.Input(shape=(32,))
    x = tf.keras.layers.Dense(1024, activation=tf.nn.selu)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.selu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(1024, activation=tf.nn.selu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.selu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def model_train(model, all_input, all_output):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='ScorePredictor.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
    )
    with tf.device('/gpu:0'):
        model.fit(x=all_input, y=all_output, batch_size=2048, epochs=10000, callbacks=[ckpt])


def model_test(model, processor=None, data_file_path='my_personality.csv', least_words=3, most_word=30):
    if processor is None:
        processor = build_processor()
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
                    embedding = embed([text_slice.lower()], processor)
                    res = model.predict(x=embedding) * 5
                    print('pred:', res.tolist()[0])
                    print('true:', score)
                    print('diff:', np.array(score) - res[0, :])
                    cmd = ''
                    while cmd not in ['y', 'n']:
                        cmd = input('continue?:y/n\n')
                    flag = (cmd == 'y')
                if not flag:
                    break
            if not flag:
                break


if __name__ == '__main__':
    train = False
    test = not train
    all_input_, all_output_ = data_load(new_data=False)
    if os.path.exists('ScorePredictor.h5'):
        model_ = tf.keras.models.load_model('ScorePredictor.h5')
    else:
        model_ = model_build()
    if train:
        model_train(model_, all_input_, all_output_)
    if test:
        model_test(model_)
