import os
import random

import numpy as np
import tensorflow as tf

from Model.BertDNN.Bert import embed, build_processor

label_dict = {
    'COM': 0,
    'PRO': 1,
    'BUS': 2,
    'NET': 3,
    'SER': 4,
    'LOC': 5,
    'ETC': 6
}
onehot_mat = np.eye(N=len(list(label_dict)))


# COM\PRO\BUS\NET\SER\LOC\ETC
# 公司、产品、商务、入网、业务、地方、其它

# n_classes = 7
def model_build(n_classes=7, force_new=False):
    if not os.path.exists('TopicsClassifier.h5') or force_new:
        inputs = tf.keras.Input(shape=(768,))
        x = tf.keras.layers.Dense(1024, activation=tf.nn.selu)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation=tf.nn.selu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(1024, activation=tf.nn.selu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation=tf.nn.selu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
        model = tf.keras.models.load_model('TopicsClassifier.h5')
    return model


def model_train(model, processor=None):
    if processor is None:
        processor = build_processor()
    x, y = data_load_topics(processor=processor)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='TopicsClassifier.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
    )
    with tf.device('/gpu:0'):
        model.fit(x=x, y=y, batch_size=2048, epochs=10, callbacks=[ckpt])


def model_test(model, processor=None):
    if processor is None:
        processor = build_processor()
    x, y = data_load_topics(processor=processor)
    flag = True
    while flag:
        index = random.choice(list(range(x.shape[0])))
        res = model.predict(x=np.expand_dims(x[index, :], axis=0))
        print('pred:', ['%.3f' % item for item in res.tolist()[0]])
        print('true:', ['%.3f' % item for item in y[index, :].tolist()])
        print('diff:', ['%.3f' % abs(item) for item in (y[index, :] - res[0, :]).tolist()])
        cmd = ''
        while cmd not in ['y', 'n']:
            cmd = input('continue?:y/n\n')
        flag = (cmd == 'y')


def data_load_topics(processor, data_path='../../Data/chat.txt'):
    f = open(data_path, 'r', encoding='utf-8')
    lines = f.readlines()
    lines = [line.split('#') for line in lines]
    x = np.array([np.squeeze(embed(input_str=line[1], masked_lm=processor)) for line in lines])
    y = np.array([onehot_mat[label_dict[line[0]], :] for line in lines])
    return x, y


if __name__ == '__main__':
    processor_ = build_processor(
        use_post_trained=False, path_post_trained='Bert.h5',
        saved_output='SavedBertEmbedding.pkl'
    )
    mdl = model_build()
    # model_train(mdl, processor_)
    model_test(mdl, processor_)
