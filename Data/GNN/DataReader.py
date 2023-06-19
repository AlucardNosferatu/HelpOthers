import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from Data.GNN.GraphBuilder import build_graph
from Data.GNN.GraphReader import read_graph
from Data.GNN.Utils import unify_word_form, get_mapper
from Data.NaiveDNN.DataReader import unify_symbol, extract_parenthesis


def read_file(start_index, vocab_size=4096, limit_text=2048, limit_author=128, mapper=None, data='my_personality.csv',
              least_words=3, most_word=30):
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
    for index in tqdm(range(start_index, data.shape[0])):
        row = data.iloc[index, :]
        text = row['STATUS'].lower()
        author = row['#AUTHID']
        s_ext = row['sEXT']
        s_neu = row['sNEU']
        s_agr = row['sAGR']
        s_con = row['sCON']
        s_opn = row['sOPN']
        score = [s_ext / 5, s_neu / 5, s_agr / 5, s_con / 5, s_opn / 5]
        if text in mapper['tlist']:
            pass
        else:
            continue
        if author in mapper['alist']:
            pass
        else:
            continue
        a_index = mapper['alist'].index(author)
        t_index = mapper['tlist'].index(text)
        t_index += len(mapper['alist'])
        text = unify_symbol(text)
        texts = extract_parenthesis(text)
        for text in texts:
            text_slices = text.split('.')
            for text_slice in text_slices:
                if least_words < len(text_slice.split(' ')) < most_word:
                    text_slice, lemmatizer, stemmer, speller = unify_word_form(text_slice, lemmatizer, stemmer, speller)
                    for word in text_slice.split(' '):
                        if word in mapper['w2i'].keys():
                            score_vec = np.array(score)
                            all_output.append(score_vec)
                            embed = [0.0] * mapper['total_dim']
                            w_index = mapper['w2i'][word] - 1
                            w_index += len(mapper['alist'])
                            w_index += len(mapper['tlist'])
                            embed[t_index] = 1.0
                            embed[a_index] = 1.0
                            embed[w_index] = 1.0
                            embed_vec = np.array(embed)
                            all_input.append(embed_vec)
                    assert len(all_input) == len(all_output)
    assert len(all_input) == len(all_output)
    time.sleep(1)
    print('数据读取完毕，总计', len(all_input), '条')
    return all_input, all_output, mapper, data


def read_data(
        vocab_size=128,
        limit_text=126,
        limit_author=2,
        start_index=0,
        data_='../my_personality.csv',
        stop_after=2048
):
    all_adj = []
    all_input = []
    all_output = []
    flag = True
    while flag:
        # 以下为训练数据生成的代码
        print('第一步：建立Batch的图')
        mapper_, data_ = build_graph(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=None, data=data_, reset=True)
        print('第二步：读取Batch范围内的数据')
        batch_input, batch_output, mapper_, data_ = read_file(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=mapper_, data=data_, least_words=3, most_word=32
        )
        print('第三步：从Batch的图读取邻接矩阵')
        sym_ama, vis_ama, mapper_, data_ = read_graph(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=mapper_, data=data_
        )
        sym_ama_list = [sym_ama for _ in batch_input]
        all_adj += sym_ama_list.copy()
        all_input += batch_input.copy()
        all_output += batch_output.copy()
        start_index = mapper_['last_index'] + 1
        # todo: 保存all_adj、all_input、all_output到文件
        #  （一个大文件比较省磁盘空间，而且IO耗时小，但是占内存）
        #  （多个小文件占用磁盘空间大，而且IO耗时也大，但是能大幅度减轻内存占用）
        if len(all_input) >= stop_after:
            flag = False
    return all_input, all_adj, all_output


if __name__ == '__main__':
    read_data(stop_after=8192)
    print('Done')
