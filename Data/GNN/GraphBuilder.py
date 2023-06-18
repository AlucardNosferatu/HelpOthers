import pandas as pd
from py2neo import Graph, Node, Relationship
from tqdm import tqdm

from Data.GNN.DataReader import get_mapper, unify_word_form
from Data.GNN.TF_IDF import tf_idf_python
from Data.NaiveDNN.DataReader import unify_symbol, extract_parenthesis


def clear_graph(graph=None):
    if graph is None:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "20291224"))
    graph.run('match (n1)-[r]-(n2) delete n1,r,n2')
    graph.run('match (n1) delete n1')


# 建立除w-w以外的关系
def write_graph(
        data='my_personality.csv',
        mapper=None,
        vocab_size=4096,
        limit_text=2048,
        limit_author=128,
        reset=True
):
    lemmatizer = None
    stemmer = None
    speller = None
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "20291224"))
    word_debug = []
    if reset:
        clear_graph(graph)
    if type(data) is str:
        data = pd.read_csv(data)
    if mapper is None:
        data, mapper = get_mapper(data, limit_author, limit_text, vocab_size)

    tf_idf, vocab = get_weights(lemmatizer, mapper, speller, stemmer)

    for index in tqdm(range(data.shape[0])):
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
                                if word not in word_debug:
                                    word_debug.append(word)
    diff = set(list(mapper['w2i'].keys())).difference(set(word_debug))
    print('Done')


def get_weights(lemmatizer, mapper, speller, stemmer):
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
    text_for_weight = text_for_weight_.copy()
    tf_idf, vocab = tf_idf_python(text_for_weight, word_all=list(mapper['w2i'].keys()))
    return tf_idf, vocab


if __name__ == '__main__':
    write_graph(
        data='../my_personality.csv',
        mapper=None,
        vocab_size=128,
        limit_text=126,
        limit_author=2,
        reset=True
    )
    print('Done')