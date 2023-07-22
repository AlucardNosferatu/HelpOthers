# feature_input = tf.keras.Input(shape=(adj_mat_dim,))
# adjacent_matrix = tf.keras.Input(shape=(adj_mat_dim, adj_mat_dim))
# x = feature_input
# x = tf.keras.layers.BatchNormalization()(x)
# x = GraphConv(num_outputs=adj_mat_dim)([x, adjacent_matrix])
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Flatten()(x)
# outputs = tf.keras.layers.Dense(score_dim, activation=tf.nn.sigmoid)(x)
# model = tf.keras.Model(inputs=[feature_input, adjacent_matrix], outputs=outputs)
import torch
import torch_geometric.nn


class GCN(torch.nn.Module):
    def __init__(self, feature_size):
        super(GCN, self).__init__()
        self.bn1 = torch.nn.LazyBatchNorm2d()
        self.gc1 = torch_geometric.nn.GCNConv(in_channels=feature_size, out_channels=256)
        self.gc2 = torch_geometric.nn.GCNConv(in_channels=256, out_channels=256)
        self.gc3 = torch_geometric.nn.GCNConv(in_channels=256, out_channels=256)
        self.gc4 = torch_geometric.nn.GCNConv(in_channels=256, out_channels=256)
        self.bn2 = torch.nn.LazyBatchNorm2d()
        self.flat=torch.nn.Flatten()
        self.dense=torch.nn.LazyLinear(out_features=5)

    def forward(self, x):
        feature_input = x[0]
        adjacent_matrix = x[1]
        x = self.bn1(feature_input)
        # GConv in torch use edge_index to build adj_mat
        # the code below is not runnable
        x = self.gc1(x, adjacent_matrix)
        x = self.gc2(x, adjacent_matrix)
        x = self.gc3(x, adjacent_matrix)
        x = self.gc4(x, adjacent_matrix)
        x = self.bn2(x)
        x = self.flat(x)
        x = self.dense(x)
        x = torch.nn.functional.sigmoid(x)
        return x
