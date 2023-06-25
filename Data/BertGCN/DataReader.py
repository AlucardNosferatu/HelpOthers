import os
import random
import threading
import time

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
        val_split=0.25,
        workers=8,
        correct_shapes=None,
):
    index_list = init_indices(gen_files_count, train_data, val_split, correct_shapes, data_folder)
    batch_size = min(len(index_list), batch_size)
    while True:
        batch_input = []
        batch_adj = []
        batch_output = []
        indices = []
        threads = []
        for _ in range(batch_size):
            choice_list = list(set(index_list).difference(set(indices)))
            index = random.choice(choice_list)
            assert index not in indices
            indices.append(index)
        read_by_threads(batch_adj, batch_input, batch_output, batch_size, data_folder, indices, threads, workers)
        batch_input = np.array(batch_input)
        batch_adj = np.array(batch_adj)
        batch_output = np.array(batch_output)
        x = [batch_input, batch_adj]
        y = batch_output
        yield x, y


def init_indices(gen_files_count, train_data, val_split, correct_shapes=None, data_folder='Batches'):
    global test_list, train_list
    if train_list is None or test_list is None:
        if correct_shapes is not None:
            print('开始校验数据格式')
            batch_validation(data_folder=data_folder, correct_shapes=correct_shapes, gen_files_count=gen_files_count)
            time.sleep(1)
            print('数据格式校验完毕')
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
    return index_list


def read_by_threads(batch_adj, batch_input, batch_output, batch_size, data_folder, indices, threads, workers):
    batch_sync = threading.Lock()
    for i in range(workers):
        indices_per_worker = [indices[j] for j in range(batch_size) if j % workers == i]
        thread = threading.Thread(
            target=read_by_thread,
            args=(batch_adj, batch_input, batch_output, data_folder, indices_per_worker.copy(), batch_sync)
        )
        threads.append(thread)
    for thread in threads:
        thread.start()
    while len(batch_input) < batch_size:
        time.sleep(0.01)


def read_by_thread(batch_adj, batch_input, batch_output, data_folder, indices, lock):
    for index in indices:
        slice_input = np.load(os.path.join(data_folder, 'Input_{}.npy'.format(index)))
        slice_adj = np.load(os.path.join(data_folder, 'AdjMat_{}.npy'.format(index)))
        slice_output = np.load(os.path.join(data_folder, 'Output_{}.npy'.format(index)))
        lock.acquire()
        batch_input.append(slice_input)
        batch_adj.append(slice_adj)
        batch_output.append(slice_output)
        lock.release()


if __name__ == '__main__':
    print('Done')
