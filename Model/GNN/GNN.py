import numpy as np
import tensorflow as tf


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
    print('Done')
