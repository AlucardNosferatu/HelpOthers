import numpy as np
import tensorflow as tf

from Data.GNN.DataReader import read_file
from Data.GNN.GraphBuilder import build_graph
from Model.GNN.GraphReader import read_graph


# Graph convolutional layer

class GraphConv(tf.keras.layers.Layer):

    def __init__(self, num_outputs, adj_mat, activation="sigmoid", **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.bias = None
        self.W = None
        self.num_outputs = num_outputs
        self.activation_function = activation
        self.adjacent_matrix = tf.Variable(adj_mat, trainable=False, dtype=tf.float32)

    def build(self, input_shape):
        # Weights
        self.W = self.add_weight("W", shape=[int(input_shape[-1]), self.num_outputs])
        # bias
        self.bias = self.add_weight("bias", shape=[self.num_outputs])

    def call(self, inputs, **kwargs):
        if self.activation_function == 'relu':
            return tf.keras.backend.relu(
                tf.keras.backend.dot(
                    tf.keras.backend.dot(self.adjacent_matrix, inputs), self.W
                ) + self.bias
            )
        else:
            return tf.keras.backend.sigmoid(
                tf.keras.backend.dot(
                    tf.keras.backend.dot(self.adjacent_matrix, inputs), self.W
                ) + self.bias
            )


if __name__ == '__main__':
    vocab_size = 128
    limit_text = 126
    limit_author = 2
    print('第一步：建立Batch的图')
    mapper_, data_ = build_graph(
        vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author, mapper=None,
        data='../../Data/my_personality.csv', reset=True
    )
    print('第二步：读取Batch范围内的数据')
    all_input, all_output, mapper_, data_ = read_file(
        vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author, mapper=mapper_,
        data=data_, least_words=3, most_word=32)
    print('第三步：从Batch的图读取邻接矩阵')
    sym_ama, vis_ama, mapper_, data_ = read_graph(
        vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author, mapper=mapper_,
        data=data_
    )
    gcn_layer = GraphConv(num_outputs=64, adj_mat=sym_ama)
    input_batch = np.copy(all_input[0, :])
    input_batch = np.expand_dims(input_batch, axis=0)
    input_batch = input_batch.transpose()
    res = gcn_layer(input_batch)
    print(res)
