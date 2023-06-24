import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from Data.TextGCN.GraphBuilder import build_graph
from Data.TextGCN.GraphReader import read_graph
from Data.TextGCN.Utils import unify_word_form, get_mapper
from Data.BertDNN.DataReader import unify_symbol, extract_parenthesis


def encoder_onehot(a_index, mapper, t_index, word):
    embed = [0.0] * mapper['total_dim']
    w_index = mapper['w2i'][word] - 1
    w_index += len(mapper['alist'])
    w_index += len(mapper['tlist'])
    embed[t_index] = 1.0
    embed[a_index] = 1.0
    embed[w_index] = 1.0
    embed_vec = np.array(embed)
    return embed_vec


def read_row(data, index):
    row = data.iloc[index, :]
    text = row['STATUS'].lower()
    author = row['#AUTHID']
    s_ext = row['sEXT']
    s_neu = row['sNEU']
    s_agr = row['sAGR']
    s_con = row['sCON']
    s_opn = row['sOPN']
    score = [s_ext / 5, s_neu / 5, s_agr / 5, s_con / 5, s_opn / 5]
    return author, score, text


def read_file(
        start_index,
        vocab_size=4096,
        limit_text=2048,
        limit_author=128,
        mapper=None,
        data='my_personality.csv',
        least_words=3,
        most_word=30,
        embed_level='word',
        embed_encoder=encoder_onehot
):
    if type(data) is str:
        data = pd.read_csv(data)
    if mapper is None:
        data, mapper = get_mapper(start_index, data, limit_author, limit_text, vocab_size)
    print('原始数据和Batch数据已载入')
    lemmatizer = None
    stemmer = None
    speller = None
    all_input = []
    all_output = []
    graph_batch = []
    prev_author = None
    prev_score = None
    for index in tqdm(range(start_index, min(start_index + limit_text, data.shape[0]))):
        author, score, text = read_row(data, index)
        if prev_author is None:
            prev_author = author
        if prev_score is None:
            prev_score = score
        if text in mapper['tlist']:
            pass
        else:
            continue
        if author in mapper['alist']:
            a_index = mapper['alist'].index(author)
        else:
            a_index = None
            if embed_level != 'graph':
                assert embed_level in ['text', 'word']
                continue
        t_index = mapper['tlist'].index(text)
        t_index += len(mapper['alist'])
        text = unify_symbol(text)
        if embed_level == 'graph':
            # matrix_order = 'author' + 'text' + 'word'

            text, lemmatizer, stemmer, speller = unify_word_form(
                text, lemmatizer, stemmer, speller
            )
            embed_vec = embed_encoder(a_index, mapper, t_index, text)
            if prev_author != author:
                assert len(graph_batch) >= mapper['total_dim']
                all_input.append(np.array(graph_batch.copy()).transpose())
                all_output.append(np.array(prev_score))
                graph_batch.clear()
            while len(graph_batch) < limit_author:
                graph_batch.append([1.0] * mapper['bert_dim'])
            while len(graph_batch) < limit_author + limit_text:
                graph_batch.append([0.0] * mapper['bert_dim'])
            while len(graph_batch) < mapper['total_dim']:
                graph_batch.append([1.0] * mapper['bert_dim'])
            graph_batch[t_index] = embed_vec
            prev_author = author
            prev_score = score
        else:
            texts = extract_parenthesis(text)
            for text in texts:
                text_slices = text.split('.')
                for text_slice in text_slices:
                    if least_words < len(text_slice.split(' ')) < most_word:
                        text_slice, lemmatizer, stemmer, speller = unify_word_form(
                            text_slice, lemmatizer, stemmer, speller
                        )
                        if embed_level == 'text':
                            score_vec = np.array(score)
                            all_output.append(score_vec)
                            embed_vec = embed_encoder(a_index, mapper, t_index, text_slice)
                            all_input.append(embed_vec)
                        elif embed_level == 'word':
                            for word in text_slice.split(' '):
                                if word in mapper['w2i'].keys():
                                    score_vec = np.array(score)
                                    all_output.append(score_vec)
                                    embed_vec = embed_encoder(a_index, mapper, t_index, word)
                                    all_input.append(embed_vec)
                                assert len(all_input) == len(all_output)
                        else:
                            raise ValueError(
                                'embed_level should be "graph", "text" or "word", get', embed_level, 'instead!'
                            )
                    assert len(all_input) == len(all_output)
        assert len(all_input) == len(all_output)
    if embed_level == 'graph':
        assert graph_batch[-vocab_size - 1] != [0.0] * mapper['total_dim']
        all_input.append(np.array(graph_batch.copy()).transpose())
        all_output.append(np.array(prev_score))
    assert len(all_input) == len(all_output)
    time.sleep(1)
    print('数据读取完毕，总计', len(all_input), '条')
    return all_input, all_output, mapper, data


def read_data(
        vocab_size=128,
        limit_text=126,
        limit_author=2,
        start_index=0,
        data='../my_personality.csv',
        stop_after=2048,
        read_file_action=read_file,
        embed_level='word',
        embed_encoder=encoder_onehot,
        save_by_batch=None
):
    all_adj = []
    all_input = []
    all_output = []
    flag = True
    batch_count = 0
    while flag:
        # 以下为训练数据生成的代码
        print('第一步：建立Batch的图')
        mapper, data = build_graph(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=None, data=data, reset=True)
        print('第二步：读取Batch范围内的数据')
        batch_input, batch_output, mapper, data = read_file_action(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=mapper, data=data, least_words=3, most_word=32, embed_level=embed_level,
            embed_encoder=embed_encoder
        )
        print('第三步：从Batch的图读取邻接矩阵')
        sym_ama, vis_ama, mapper, data = read_graph(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=mapper, data=data
        )
        sym_ama_list = [sym_ama for _ in batch_input]
        batch_start_index = batch_count
        batch_count += len(batch_input)
        # todo: 保存all_adj、all_input、all_output到文件
        #  （一个大文件比较省磁盘空间，而且IO耗时小，但是占内存）
        #  （多个小文件占用磁盘空间大，而且IO耗时也大，但是能大幅度减轻内存占用）
        if save_by_batch is not None:
            for i in range(len(batch_input)):
                np.save(
                    os.path.join(save_by_batch, 'AdjMat_{}.npy'.format(batch_start_index + i)),
                    np.array(sym_ama_list[i])
                )
                np.save(
                    os.path.join(save_by_batch, 'Input_{}.npy'.format(batch_start_index + i)),
                    np.array(batch_input[i])
                )
                np.save(
                    os.path.join(save_by_batch, 'Output_{}.npy'.format(batch_start_index + i)),
                    np.array(batch_output[i])
                )
        else:
            all_adj += sym_ama_list.copy()
            all_input += batch_input.copy()
            all_output += batch_output.copy()
        start_index = mapper['last_index'] + 1
        if batch_count >= stop_after:
            flag = False
        time.sleep(1)
        print('数据已采集：', batch_count, '/', stop_after)
    return all_input, all_adj, all_output


if __name__ == '__main__':
    read_data(stop_after=8192)
    print('Done')
