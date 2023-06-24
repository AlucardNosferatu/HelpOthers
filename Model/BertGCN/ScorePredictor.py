import os

import tensorflow as tf

from Data.BertGCN.DataReader import encoder_bert
from Model.TextGCN.GNN import GraphConv
from Model.TextGCN.ScorePredictor import data_load, model_build, model_train, model_test

if __name__ == '__main__':
    train = True
    test = not train
    all_input_, all_adj_, all_output_ = data_load(
        new_data=True,
        embed_level='graph',
        embed_encoder=encoder_bert,
        data_folder='BertGCN',
        save_by_batch='../../Data/BertGCN'
    )
    if os.path.exists('ScorePredictor.h5'):
        model_ = tf.keras.models.load_model('ScorePredictor.h5', custom_objects={'GraphConv': GraphConv})
    else:
        model_ = model_build(feature_input=tf.keras.Input(shape=(256, None)))
    if train:
        model_train(model_, all_input_, all_adj_, all_output_)
    if test:
        model_test(model_, all_input_, all_adj_, all_output_)
