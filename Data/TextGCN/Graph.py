import time

import numpy as np
import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from tqdm import tqdm

from Data.TextGCN.Utils import unify_word_form, get_mapper
from Data.TextGCN.PPMI import pmi
from Data.TextGCN.TF_IDF import tf_idf_python
from Data.BertDNN.DataReader import unify_symbol, extract_parenthesis


def clear_graph(graph=None):
    if graph is None:
        # GUOER: use 'bolt://server.natappfree.cc:42148' auth=('GUOER','Your QQ Number')
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "20291224"))
    graph.run('match (n1)-[r]-(n2) delete n1,r,n2')
    graph.run('match (n1) delete n1')


# 建立除w-w以外的关系
def read_graph(start_index, vocab_size=4096, limit_text=2048, limit_author=128, mapper=None, data='my_personality.csv',
               reset=True, bert_dim=8, path_post_trained='../../Model/BertDNN/Bert.h5', saved_output=None):
    if type(data) is str:
        data = pd.read_csv(data)
    if mapper is None:
        data, mapper = get_mapper(start_index, data, limit_author, limit_text, vocab_size, bert_dim, path_post_trained,
                                  saved_output)
    print('原始数据和Batch数据已载入')
    word2index = mapper['w2i']
    index2word = mapper['i2w']
    text_count_list = mapper['tlist']
    author_count_list = mapper['alist']
    total_dim = mapper['total_dim']
    lemmatizer = None
    stemmer = None
    speller = None
    # GUOER: use 'bolt://server.natappfree.cc:42148' auth=('GUOER','Your QQ Number')
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "20291224"))
    if reset:
        clear_graph(graph)
        print('图数据库已清空')
    tf_idf, vocab, pmi_pairs = get_weights(mapper, lemmatizer, speller, stemmer)
    print('TF-IDF和PMI计算完成')
    print('开始建立作者-文档、文档-词汇关系')
    for index in tqdm(range(start_index, min(start_index + limit_text, data.shape[0]))):
        row = data.iloc[index, :]
        text = row['STATUS'].lower()
        author = row['#AUTHID']
        if text in text_count_list:
            t_index = text_count_list.index(text)
            text_node = Node("Text", name=str(t_index), content=text)
            graph.merge(text_node, 'Text', 'name')
        else:
            t_index = None
            text_node = None
        if author in author_count_list:
            a_index = author_count_list.index(author)
            author_node = Node("Author", name=str(a_index), content=author)
            graph.merge(author_node, 'Author', 'name')
        else:
            author_node = None
        if None not in [author_node, text_node]:
            weight = {'value': "1"}
            at = Relationship(author_node, "write", text_node, **weight)
            graph.merge(at, 'Author', 'name')
        if text_node is not None:
            text = unify_symbol(text)
            texts = extract_parenthesis(text)
            for text in texts:
                text_slices = text.split('.')
                for text_slice in text_slices:
                    text_slice, lemmatizer, stemmer, speller = unify_word_form(
                        text_slice,
                        lemmatizer,
                        stemmer,
                        speller
                    )
                    for word in text_slice.split(' '):
                        if word in word2index.keys():
                            if t_index is not None:
                                word_node = Node("Word", name=word)
                                graph.merge(word_node, 'Word', 'name')
                                tf_idf_w_in_t = tf_idf[t_index][vocab.index(word)]
                                weight = {'value': tf_idf_w_in_t}
                                wt = Relationship(word_node, "in", text_node, **weight)
                                graph.merge(wt, 'Word', 'name')
    time.sleep(1)
    print('作者-文档、文档-词汇关系建立完毕')
    matcher_node = NodeMatcher(graph)
    for pmi_pair in tqdm(pmi_pairs):
        word_pair = pmi_pair[0]
        word1 = word_pair[0]
        word2 = word_pair[1]
        pmi_ = pmi_pair[1]
        node1 = matcher_node.match("Word", name=word1).first()
        node2 = matcher_node.match("Word", name=word2).first()
        weight = {'value': pmi_}
        ww = Relationship(node1, "near", node2, **weight)
        graph.merge(ww, 'Word', 'name')
    time.sleep(1)
    print('词汇-词汇关系建立完毕')

    # matrix_order = 'author' + 'text' + 'word'
    author_offset = 0
    text_offset = len(author_count_list) + author_offset
    word_offset = len(text_count_list) + text_offset

    adj_matrix = []
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
    while len(adj_matrix) < total_dim - vocab_size:
        adj_matrix.append([0.0] * total_dim)
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
    return mapper, data, sym_ama, vis_ama


def get_weights(mapper, lemmatizer, speller, stemmer):
    print('汇总并格式化语料')
    text_for_weight = mapper['tlist'].copy()
    text_for_weight = [unify_symbol(item) for item in text_for_weight]
    text_for_weight_ = []
    for item in text_for_weight:
        text_for_weight_.append(' '.join(extract_parenthesis(item)).replace('.', ' '))
    text_for_weight = text_for_weight_.copy()
    text_for_weight_.clear()
    for item in tqdm(text_for_weight):
        item, lemmatizer, stemmer, speller = unify_word_form(
            item,
            lemmatizer,
            stemmer,
            speller
        )
        text_for_weight_.append(item)
    time.sleep(1)
    print('语料已汇总')
    text_for_weight = text_for_weight_.copy()
    tf_idf, vocab = tf_idf_python(text_for_weight, word_all=list(mapper['w2i'].keys()))
    print('TF-IDF已计算')
    pmi_pairs = pmi(text=' '.join(text_for_weight), vocab=vocab)
    print('PMI已计算')
    return tf_idf, vocab, pmi_pairs


def read_graph_loop(
        adj_mat_list, bert_dim, data, limit_author, limit_text, mapper_list, path_post_trained, saved_output,
        start_index, vocab_size, t_lock, status
):
    while True:
        mapper, data, sym_ama, vis_ama = read_graph(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=None, data=data, reset=True, bert_dim=bert_dim, path_post_trained=path_post_trained,
            saved_output=saved_output
        )
        start_index = mapper['last_index'] + 1
        t_lock.acquire()
        mapper_list.append(mapper)
        adj_mat_list.append(sym_ama)
        t_lock.release()
        if len(status) <= 0:
            status.append('RGL_START')
        if start_index >= data.shape[0]:
            status[0] = 'RGL_FINISH'
            break


if __name__ == '__main__':
    print('Done')
