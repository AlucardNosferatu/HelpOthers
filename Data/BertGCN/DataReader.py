from Data.TextGCN.DataReader import read_data
from Model.BertDNN.Bert import build_processor, tokenize

processor = None


def encoder_bert(a_index, mapper, t_index, graph):
    global processor
    if processor is None:
        processor = build_processor(seq_len=mapper['total_dim'])
    text_vec = tokenize(graph, processor)
    text_vec = text_vec.astype(float)
    _ = a_index
    _ = t_index
    return text_vec.tolist()


if __name__ == '__main__':
    all_input, all_adj, all_output = read_data(stop_after=64, embed_level='graph', embed_encoder=encoder_bert)
    print('Done')
