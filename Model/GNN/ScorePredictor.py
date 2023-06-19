import os.path
import random

import numpy as np
import tensorflow as tf

from Data.GNN.DataReader import read_data
from Data.NaiveDNN.DataReader import unify_symbol, extract_parenthesis
from Model.GNN.GNN import GraphConv


def data_load(new_data=False):
    if new_data:
        all_input, all_adj, all_output = read_data()
        np.save('../../Data/GNN/Input.npy', np.array(all_input))
        np.save('../../Data/GNN/Output.npy', np.array(all_output))
        np.save('../../Data/GNN/AdjMat.npy', np.array(all_adj))
    else:
        all_input = np.load('../../Data/GNN/Input.npy')
        all_output = np.load('../../Data/GNN/Output.npy')
        all_adj = np.load('../../Data/GNN/AdjMat.npy')
    return all_input, all_adj, all_output


def model_build():
    feature_input = tf.keras.Input(shape=(256,))
    adjacent_matrix = tf.keras.Input(shape=(256, 256))
    x = GraphConv(num_outputs=256, activation='relu')([feature_input, adjacent_matrix])
    x = tf.keras.layers.Dense(512, activation=tf.nn.selu)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.selu)(x)
    x = GraphConv(num_outputs=256, activation='relu')([x, adjacent_matrix])
    x = tf.keras.layers.Dense(512, activation=tf.nn.selu)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.selu)(x)
    x = GraphConv(num_outputs=256, activation='relu')([x, adjacent_matrix])
    x = tf.keras.layers.Dense(512, activation=tf.nn.selu)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.selu)(x)
    x = GraphConv(num_outputs=256, activation='relu')([x, adjacent_matrix])
    x = tf.keras.layers.Dense(512, activation=tf.nn.selu)(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.selu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs=[feature_input, adjacent_matrix], outputs=outputs)
    return model


def model_train(model, all_input, all_adj, all_output):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-7),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
        run_eagerly=True
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='ScorePredictor.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
    )
    tf.keras.utils.plot_model(model, 'GCN.png', show_shapes=True, expand_nested=True, show_layer_activations=True)
    with tf.device('/cpu:0'):
        model.fit(x=[all_input, all_adj], y=all_output, batch_size=2048, epochs=10000, callbacks=[ckpt], shuffle=True)


def model_test(model, all_input, all_adj, all_output):
    flag = True
    while flag:
        index = random.choice(list(range(len(all_input))))
        text_feature = np.expand_dims(all_input[index], axis=0)
        adjacent_mat = np.expand_dims(all_adj[index], axis=0)
        score = [score * 5 for score in all_output[index]]
        res = model.predict([text_feature, adjacent_mat]) * 5
        print('pred:', res.tolist()[0])
        print('true:', score)
        print('diff:', np.array(score) - res[0, :])
        cmd = ''
        while cmd not in ['y', 'n']:
            cmd = input('continue?:y/n\n')
        flag = (cmd == 'y')


if __name__ == '__main__':
    train = False
    test = not train
    all_input_, all_adj_, all_output_ = data_load(new_data=False)
    if os.path.exists('ScorePredictor.h5'):
        model_ = tf.keras.models.load_model('ScorePredictor.h5', custom_objects={'GraphConv': GraphConv})
    else:
        model_ = model_build()
    if train:
        model_train(model_, all_input_, all_adj_, all_output_)
    if test:
        model_test(model_, all_input_, all_adj_, all_output_)
