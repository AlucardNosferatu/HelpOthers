import numpy as np
import tensorflow as tf

from Data.GNN.DataReader import read_file


# Graph convolutional layer

class GraphConv(tf.keras.layers.Layer):

    def __init__(self, num_outputs, adj_mat, activation="sigmoid", **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.bias = None
        self.W = None
        self.num_outputs = num_outputs
        self.activation_function = activation
        self.adjacent_matrix = tf.Variable(adj_mat, trainable=False)

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
    all_input, all_output = read_file(
        data='../../Data/my_personality.csv',
        mapper=None,
        least_words=3,
        most_word=30,
        vocab_size=128,
        limit_text=64,
        limit_author=64
    )
    adj = np.eye(256, dtype=np.float32)
    gcn_layer = GraphConv(num_outputs=64, adj_mat=adj)
    res = gcn_layer(np.expand_dims(all_input[0], axis=0).transpose())
