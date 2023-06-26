import os.path
import random

import numpy as np
import tensorflow as tf

from Data.BertGCN.DataReader import batch_generator
from Data.TextGCN.DataReader import read_data, encoder_onehot
from Model.TextGCN.GNN import GraphConv


def data_load(
        new_data=False,
        stop_after=8192,
        embed_level='word',
        embed_encoder=encoder_onehot,
        data_folder='TextGCN',
        save_by_batch=False,
        bert_dim=8,
        binary_label=False,
        start_index=0,
        path_post_trained='../BertDNN/Bert.h5'
):
    if new_data:
        if save_by_batch:
            batch_saving_dir = '../../Data/{}/Batches'.format(data_folder)
        else:
            batch_saving_dir = None
        all_input, all_adj, all_output = read_data(
            data='../../Data/my_personality.csv',
            stop_after=stop_after,
            embed_level=embed_level,
            embed_encoder=embed_encoder,
            save_by_batch=batch_saving_dir,
            bert_dim=bert_dim,
            binary_label=binary_label,
            start_index=start_index,
            path_post_trained=path_post_trained
        )
        if not save_by_batch:
            all_input = np.array(all_input)
            all_output = np.array(all_output)
            all_adj = np.array(all_adj)
            np.save('../../Data/{}/Input.npy'.format(data_folder), all_input)
            np.save('../../Data/{}/Output.npy'.format(data_folder), all_output)
            np.save('../../Data/{}/AdjMat.npy'.format(data_folder), all_adj)
    if save_by_batch:
        # todo: batch generator ([[input,adj],output])
        all_input = batch_generator
        all_output = None
        all_adj = None
    else:
        all_input = np.load('../../Data/{}/Input.npy'.format(data_folder))
        all_output = np.load('../../Data/{}/Output.npy'.format(data_folder))
        all_adj = np.load('../../Data/{}/AdjMat.npy'.format(data_folder))
    return all_input, all_adj, all_output


def model_build(feature_input=None):
    if feature_input is None:
        feature_input = tf.keras.Input(shape=(256,))
    adjacent_matrix = tf.keras.Input(shape=(256, 256))

    x = GraphConv(num_outputs=512, activation='relu')([feature_input, adjacent_matrix])
    x = tf.keras.layers.Dense(2048, activation=tf.nn.selu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.selu)(x)
    x = GraphConv(num_outputs=512, activation='relu')([x, adjacent_matrix])
    x = tf.keras.layers.Dense(2048, activation=tf.nn.selu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.selu)(x)
    x = GraphConv(num_outputs=512, activation='relu')([x, adjacent_matrix])
    x = tf.keras.layers.Dense(2048, activation=tf.nn.selu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.selu)(x)
    x = GraphConv(num_outputs=512, activation='relu')([x, adjacent_matrix])
    x = tf.keras.layers.Dense(2048, activation=tf.nn.selu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation=tf.nn.selu)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs=[feature_input, adjacent_matrix], outputs=outputs)
    return model


def model_train(
        model, all_input, all_adj, all_output, use_generator=False, gen_files_count=8192, batch_size=2048,
        step_per_epoch=100, val_split=0.25, workers=8
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-7),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'],
        run_eagerly=True
    )
    print('模型编译完成')
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='ScorePredictor.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
    )
    with tf.device('/gpu:0'):
        if use_generator:
            print('开始创建训练/测试数据生成器')
            shapes = [
                (32, 256),
                (256, 256),
                (5,)
            ]
            gen_train = all_input(
                batch_size=batch_size,
                gen_files_count=gen_files_count,
                train_data=True,
                val_split=val_split,
                workers=workers,
                correct_shapes=shapes
            )
            gen_test = all_input(
                batch_size=batch_size,
                gen_files_count=gen_files_count,
                train_data=False,
                workers=workers
            )
            print('训练/测试数据生成器创建完毕')
            print('现在开始训练')
            model.fit(
                x=gen_train,
                epochs=10000,
                callbacks=[ckpt],
                shuffle=True,
                steps_per_epoch=step_per_epoch,
                validation_data=gen_test,
                validation_steps=int(step_per_epoch * val_split)
            )
        else:
            print('现在开始训练')
            model.fit(
                x=[all_input, all_adj],
                y=all_output,
                batch_size=batch_size,
                epochs=10000,
                callbacks=[ckpt],
                shuffle=True,
                validation_split=val_split
            )


def model_test(model, all_input, all_adj, all_output, use_generator=False, gen_files_count=8192):
    def form_str(score_list):
        str_list = []
        for val in score_list:
            if val > 0:
                str_list.append('+{:.4f}'.format(val))
            else:
                str_list.append('{:.4f}'.format(val))
        return str_list

    flag = True
    while flag:
        if use_generator:
            gen = all_input(batch_size=1, gen_files_count=gen_files_count)
            x, y = gen.__next__()
            text_feature = x[0]
            adjacent_mat = x[1]
            score = (y[0] * 5).tolist()
        else:
            index = random.choice(list(range(len(all_input))))
            text_feature = np.expand_dims(all_input[index], axis=0)
            adjacent_mat = np.expand_dims(all_adj[index], axis=0)
            score = [score * 5 for score in all_output[index]]
        res = model.predict([text_feature, adjacent_mat]) * 5
        pred = form_str(res.tolist()[0])
        true = form_str(score)
        diff = form_str((np.array(score) - res[0, :]).tolist())
        print('pred:', pred)
        print('true:', true)
        print('diff:', diff)
        cmd = ''
        while cmd not in ['y', 'n']:
            cmd = input('continue?:y/n\n')
        flag = (cmd == 'y')


if __name__ == '__main__':
    train = False
    test = not train
    all_input_, all_adj_, all_output_ = data_load(new_data=False, binary_label=False)
    if os.path.exists('ScorePredictor.h5'):
        model_ = tf.keras.models.load_model('ScorePredictor.h5', custom_objects={'GraphConv': GraphConv})
    else:
        model_ = model_build()
    tf.keras.utils.plot_model(model_, 'TextGCN.png', show_shapes=True, expand_nested=True, show_layer_activations=True)
    if train:
        model_train(model_, all_input_, all_adj_, all_output_)
    if test:
        model_test(model_, all_input_, all_adj_, all_output_)
