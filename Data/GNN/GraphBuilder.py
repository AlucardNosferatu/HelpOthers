import time

import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher
from tqdm import tqdm

from Data.GNN.Utils import unify_word_form, get_mapper
from Data.GNN.PPMI import pmi
from Data.GNN.TF_IDF import tf_idf_python
from Data.NaiveDNN.DataReader import unify_symbol, extract_parenthesis


def clear_graph(graph=None):
    if graph is None:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "20291224"))
    graph.run('match (n1)-[r]-(n2) delete n1,r,n2')
    graph.run('match (n1) delete n1')


# 建立除w-w以外的关系
def build_graph(start_index, vocab_size=4096, limit_text=2048, limit_author=128, mapper=None, data='my_personality.csv',
                reset=True):
    if type(data) is str:
        data = pd.read_csv(data)
    if mapper is None:
        data, mapper = get_mapper(start_index, data, limit_author, limit_text, vocab_size)
    print('原始数据和Batch数据已载入')
    lemmatizer = None
    stemmer = None
    speller = None
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "20291224"))
    if reset:
        clear_graph(graph)
        print('图数据库已清空')
    tf_idf, vocab, pmi_pairs = get_weights(mapper, lemmatizer, speller, stemmer)
    print('TF-IDF和PMI计算完成')
    print('开始建立作者-文档、文档-词汇关系')
    for index in tqdm(range(start_index, data.shape[0])):
        row = data.iloc[index, :]
        text = row['STATUS'].lower()
        author = row['#AUTHID']
        if text in mapper['tlist']:
            t_index = mapper['tlist'].index(text)
            text_node = Node("Text", name=str(t_index), content=text)
            graph.merge(text_node, 'Text', 'name')
        else:
            t_index = None
            text_node = None
        if author in mapper['alist']:
            a_index = mapper['alist'].index(author)
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
                        if word in mapper['w2i'].keys():
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
    return mapper, data


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


if __name__ == '__main__':
    print('Done')
