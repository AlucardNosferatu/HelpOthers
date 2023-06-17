import pandas as pd
from py2neo import Graph, Node, Relationship
from tqdm import tqdm

from Data.GNN.DataReader import get_mapper
from Data.NaiveDNN.DataReader import unify_symbol, extract_parenthesis


def clear_graph(graph=None):
    if graph is None:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "20291224"))
    graph.run('match (n1)-[r]-(n2) delete n1,r,n2')


# 建立除w-w以外的关系
def write_graph(
        data='my_personality.csv',
        mapper=None,
        least_words=3,
        most_word=30,
        vocab_size=4096,
        limit_text=2048,
        limit_author=128,
        reset=True
):
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "20291224"))
    if reset:
        clear_graph(graph)
    if type(data) is str:
        data = pd.read_csv(data)
    if mapper is None:
        data, mapper = get_mapper(data, limit_author, limit_text, vocab_size)
    for index in tqdm(range(data.shape[0])):
        row = data.iloc[index, :]
        text = row['STATUS']
        author = row['#AUTHID']
        if text in mapper['tlist']:
            pass
        else:
            continue
        if author in mapper['alist']:
            pass
        else:
            continue
        t_index = mapper['tlist'].index(text)
        text_node = Node("Text", name=str(t_index), content=text)
        graph.merge(text_node)
        a_index = mapper['alist'].index(author)
        author_node = Node("Author", name=str(a_index), content=author)
        graph.merge(author_node)
        weight = {'value': "1"}
        at = Relationship(author_node, "write", text_node, **weight)
        graph.merge(at)
        text = unify_symbol(text)
        texts = extract_parenthesis(text)
        for text in texts:
            text_slices = text.split('.')
            for text_slice in text_slices:
                if least_words < len(text_slice.split(' ')) < most_word:
                    for word in text_slice.split(' '):
                        if word in mapper['w2i'].keys():
                            word_node = Node("Word", name="word")
                            graph.merge(word_node)
                            weight = {'value': "tf-idf"}
                            wt = Relationship(word_node, "in", text_node, **weight)
                            graph.merge(wt)


if __name__ == '__main__':
    write_graph(
        data='../my_personality.csv',
        mapper=None,
        least_words=3,
        most_word=30,
        vocab_size=128,
        limit_text=96,
        limit_author=32,
        reset=True
    )
