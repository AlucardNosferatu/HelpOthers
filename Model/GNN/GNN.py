import numpy as np
import tensorflow as tf

from Data.GNN.DataReader import read_file
from Data.GNN.GraphBuilder import build_graph
from Model.GNN.GraphReader import read_graph


# Graph convolutional layer

class GraphConv(tf.keras.layers.Layer):

    def __init__(self, num_outputs, activation="sigmoid", **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.bias = None
        self.W = None
        self.num_outputs = num_outputs
        self.activation_function = activation

    def build(self, input_shape):
        # Weights
        self.W = self.add_weight("W", shape=[int(input_shape[0][-1]), self.num_outputs])
        # bias
        self.bias = self.add_weight("bias", shape=[self.num_outputs])

    def call(self, inputs, **kwargs):
        node_feature = inputs[0]
        adj_mat = inputs[1]
        if self.activation_function == 'relu':
            return tf.keras.backend.relu(
                tf.keras.backend.dot(
                    tf.keras.backend.dot(adj_mat, node_feature), self.W
                ) + self.bias
            )
        else:
            return tf.keras.backend.sigmoid(
                tf.keras.backend.dot(
                    tf.keras.backend.dot(adj_mat, node_feature), self.W
                ) + self.bias
            )


def gcn_test(input_vector_list, adjacent_matrix):
    # 以下为测试GCN计算正确性的代码
    gcn_layer = GraphConv(num_outputs=64)
    input_batch = np.copy(input_vector_list[0])
    input_batch = np.expand_dims(input_batch, axis=0)
    input_batch = input_batch.transpose()
    res = gcn_layer([input_batch, adjacent_matrix])
    print(res)


if __name__ == '__main__':
    vocab_size = 128
    limit_text = 126
    limit_author = 2
    start_index = 0
    data_ = '../../Data/my_personality.csv'
    all_adj = []
    all_input = []
    all_output = []
    flag = True
    while flag:
        # 以下为训练数据生成的代码
        print('第一步：建立Batch的图')
        mapper_, data_ = build_graph(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=None, data=data_, reset=True)
        print('第二步：读取Batch范围内的数据')
        batch_input, batch_output, mapper_, data_ = read_file(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=mapper_, data=data_, least_words=3, most_word=32
        )
        print('第三步：从Batch的图读取邻接矩阵')
        sym_ama, vis_ama, mapper_, data_ = read_graph(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=mapper_, data=data_
        )
        sym_ama_list = [sym_ama for _ in batch_input]
        all_adj += sym_ama_list.copy()
        all_input += batch_input.copy()
        all_output += batch_output.copy()
        start_index = mapper_['last_index'] + 1
        # todo: 保存all_adj、all_input、all_output到文件
        #  （一个大文件比较省磁盘空间，而且IO耗时小，但是占内存）
        #  （多个小文件占用磁盘空间大，而且IO耗时也大，但是能大幅度减轻内存占用）
        if len(all_input) >= 2048:
            flag = False
    print('Done')
