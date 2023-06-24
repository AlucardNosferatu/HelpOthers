import os

import tensorflow as tf

from Data.BertGCN.DataReader import encoder_bert
from Model.TextGCN.GNN import GraphConv
from Model.TextGCN.ScorePredictor import data_load, model_build, model_train, model_test

if __name__ == '__main__':
    train = True
    test = not train
    all_input_, _, _ = data_load(
        new_data=False,
        stop_after=16,
        embed_level='graph',
        embed_encoder=encoder_bert,
        data_folder='BertGCN',
        save_by_batch=True
    )
    if os.path.exists('ScorePredictor.h5'):
        model_ = tf.keras.models.load_model('ScorePredictor.h5', custom_objects={'GraphConv': GraphConv})
    else:
        model_ = model_build(feature_input=tf.keras.Input(shape=(8, 256)))
    if train:
        model_train(model_, all_input_, None, None, use_generator=True, batch_size=8, gen_files_count=16)
    if test:
        model_test(model_, all_input_, None, None, use_generator=True, gen_files_count=8192)
