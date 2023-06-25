import os
import random

import numpy as np
from tqdm import tqdm

from Model.BertDNN.Bert import build_processor, tokenize

processor = None
train_list = None
test_list = None


def encoder_bert(a_index, mapper, t_index, graph):
    global processor
    if processor is None:
        processor = build_processor(seq_len=mapper['bert_dim'])
    text_vec = tokenize(graph, processor)
    text_vec = text_vec.astype(float)
    _ = a_index
    _ = t_index
    return text_vec.tolist()


def batch_validation(correct_shapes, gen_files_count=8192, data_folder='Batches'):
    input_shape = correct_shapes[0]
    adj_shape = correct_shapes[1]
    output_shape = correct_shapes[2]
    for i in tqdm(range(gen_files_count)):
        slice_input = np.load(os.path.join(data_folder, 'Input_{}.npy'.format(i)))
        assert slice_input.shape == input_shape
        slice_adj = np.load(os.path.join(data_folder, 'AdjMat_{}.npy'.format(i)))
        assert slice_adj.shape == adj_shape
        slice_output = np.load(os.path.join(data_folder, 'Output_{}.npy'.format(i)))
        assert slice_output.shape == output_shape


def batch_generator(
        batch_size=8,
        gen_files_count=8192,
        data_folder='../../Data/BertGCN/Batches',
        train_data=True,
        val_split=0.25
):
    global train_list, test_list
    if train_list is None or test_list is None:
        all_list = list(range(gen_files_count))
        test_list = []
        while len(test_list) < (val_split * gen_files_count):
            index = random.choice(all_list)
            while index in test_list:
                index = random.choice(all_list)
            test_list.append(index)
        train_list = list(set(all_list).difference(set(test_list)))
    if train_data:
        index_list = train_list
    else:
        index_list = test_list
    while True:
        batch_input = []
        batch_adj = []
        batch_output = []
        indices = []
        while len(batch_input) < batch_size:
            index = random.choice(index_list)
            while index in indices:
                index = random.choice(index_list)
            slice_input = np.load(os.path.join(data_folder, 'Input_{}.npy'.format(index)))
            slice_adj = np.load(os.path.join(data_folder, 'AdjMat_{}.npy'.format(index)))
            slice_output = np.load(os.path.join(data_folder, 'Output_{}.npy'.format(index)))
            batch_input.append(slice_input)
            batch_adj.append(slice_adj)
            batch_output.append(slice_output)
            indices.append(index)
        batch_input = np.array(batch_input)
        batch_adj = np.array(batch_adj)
        batch_output = np.array(batch_output)
        x = [batch_input, batch_adj]
        y = batch_output
        yield x, y


if __name__ == '__main__':
    shapes = [
        (32, 256),
        (256, 256),
        (5,)
    ]
    batch_validation(correct_shapes=shapes, gen_files_count=325)
    print('Done')
