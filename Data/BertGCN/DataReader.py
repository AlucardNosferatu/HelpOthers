import os
import random

import numpy as np

from Model.BertDNN.Bert import build_processor, tokenize

processor = None


def encoder_bert(a_index, mapper, t_index, graph):
    global processor
    if processor is None:
        processor = build_processor(seq_len=mapper['total_dim'])
    text_vec = tokenize(graph, processor)
    text_vec = text_vec.astype(float)
    _ = a_index
    _ = t_index
    return text_vec.tolist()


def batch_generator(batch_size=8, gen_files_count=8192, data_folder='../../Data/BertGCN/Batches'):
    index_list = list(range(gen_files_count))
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
    print('Done')
