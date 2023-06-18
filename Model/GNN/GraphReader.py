import time

import cv2
import numpy as np
import pandas as pd
from py2neo import Graph, NodeMatcher, RelationshipMatcher
from tqdm import tqdm

from Data.GNN.DataReader import get_mapper


def read_graph(vocab_size, limit_text, limit_author, mapper=None, data='my_personality.csv'):
    if type(data) is str:
        data = pd.read_csv(data)
    if mapper is None:
        data, mapper = get_mapper(data, limit_author, limit_text, vocab_size)
    print('原始数据和Batch数据已载入')
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "20291224"))
    word2index = mapper['w2i']
    index2word = mapper['i2w']
    text_count_list = mapper['tlist']
    author_count_list = mapper['alist']
    total_dim = mapper['total_dim']
    # matrix_order = 'author' + 'text' + 'word'
    author_offset = 0
    text_offset = len(author_count_list) + author_offset
    word_offset = len(text_count_list) + text_offset
    adj_matrix = []
    matcher_node = NodeMatcher(graph)
    matcher_rel = RelationshipMatcher(graph)
    print('开始写入作者-文档关系到邻接矩阵')
    for i in tqdm(range(len(author_count_list))):
        adj_row = [0.0] * total_dim
        adj_row[i] = 1.0
        match_val = str(i)
        node = matcher_node.match("Author", name=match_val).first()
        texts = matcher_rel.match([node], r_type='write').all()
        for text in texts:
            text_index = int(text.end_node['name'])
            adj_row[text_index + text_offset] = float(text['value'])
        adj_matrix.append(adj_row)
    time.sleep(1)
    print('作者-文档关系已写入邻接矩阵')
    print('开始写入文档-词汇关系到邻接矩阵')
    for i in tqdm(range(len(text_count_list))):
        adj_row = [0.0] * total_dim
        adj_row[i + text_offset] = 1.0
        match_val = str(i)
        node = matcher_node.match("Text", name=match_val).first()
        words = matcher_rel.match([None, node], r_type='in').all()
        for word in words:
            word_str = word.start_node['name']
            word_index = word2index[word_str] - 1
            adj_row[word_index + word_offset] = float(word['value'])
        adj_matrix.append(adj_row)
    time.sleep(1)
    print('文档-词汇关系已写入邻接矩阵')
    print('开始写入词汇-词汇关系到邻接矩阵')
    for i in tqdm(range(len(list(word2index.keys())))):
        adj_row = [0.0] * total_dim
        adj_row[i + word_offset] = 1.0
        match_val = index2word[i + 1]
        node = matcher_node.match("Word", name=match_val).first()
        other_words = matcher_rel.match([node], r_type='near').all()
        for word in other_words:
            word_str = word.end_node['name']
            word_index = word2index[word_str] - 1
            adj_row[word_index + word_offset] = float(word['value'])
        adj_matrix.append(adj_row)
    time.sleep(1)
    print('词汇-词汇关系已写入邻接矩阵，邻接矩阵已装填')
    adj_matrix_array = np.array(adj_matrix)
    sym_ama = adj_matrix_array + adj_matrix_array.transpose() - np.eye(total_dim)
    print('邻接矩阵对称化已完成')
    vis_ama = np.copy(sym_ama)
    vis_ama -= np.min(vis_ama)
    vis_ama /= np.max(vis_ama)
    return sym_ama, vis_ama, mapper, data


if __name__ == '__main__':
    print('Done')
