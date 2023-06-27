import os
import threading
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from Data.BertDNN.DataReader import unify_symbol, extract_parenthesis
from Data.TextGCN.Graph import read_graph, read_graph_loop
from Data.TextGCN.Utils import unify_word_form, get_mapper, batch_rename

lock = threading.Lock()


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


def read_row(data, index, binary_label=False):
    row = data.iloc[index, :]
    text = row['STATUS'].lower()
    author = row['#AUTHID']
    if binary_label:
        score_dict = {'y': 1.0, 'n': 0.0}
        c_ext = score_dict[row['cEXT']]
        c_neu = score_dict[row['cNEU']]
        c_agr = score_dict[row['cAGR']]
        c_con = score_dict[row['cCON']]
        c_opn = score_dict[row['cOPN']]
        score = [c_ext, c_neu, c_agr, c_con, c_opn]
    else:
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
        embed_encoder=encoder_onehot,
        bert_dim=8,
        binary_label=False,
        path_post_trained='../../Model/BertDNN/Bert.h5'
):
    if type(data) is str:
        data = pd.read_csv(data)
    if mapper is None:
        data, mapper = get_mapper(start_index, data, limit_author, limit_text, vocab_size, bert_dim, path_post_trained)
    print('原始数据和Batch数据已载入')
    lemmatizer = None
    stemmer = None
    speller = None
    all_input = []
    all_output = []
    graph_batch = []
    prev_author = None
    prev_score = None
    display_pbar = False
    progress = range(start_index, min(start_index + limit_text, data.shape[0]))
    if display_pbar:
        progress = tqdm(progress)
    for index in progress:
        author, score, text = read_row(data, index, binary_label)
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


def read_data_adapter(new=False, **kwargs):
    if new:
        read_data_new(**kwargs)
        all_input, all_adj, all_output = None, None, None
    else:
        all_input, all_adj, all_output = read_data_old(**kwargs)
    return all_input, all_adj, all_output


def read_data_old(
        vocab_size=128, limit_text=126, limit_author=2,
        start_index=0,
        data='../my_personality.csv',
        stop_after=2048,
        read_file_action=read_file,
        embed_level='word', embed_encoder=encoder_onehot,
        save_by_batch=None,
        bert_dim=8,
        binary_label=False,
        path_post_trained='../../Model/BertDNN/Bert.h5',
        batch_count=0,
        saved_output=None
):
    all_adj = []
    all_input = []
    all_output = []
    flag = True
    while flag:
        # 以下为训练数据生成的代码
        print('第一步：建立Batch的图并读取邻接矩阵')
        mapper, data, sym_ama, vis_ama = read_graph(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=None, data=data, reset=True, bert_dim=bert_dim, path_post_trained=path_post_trained,
            saved_output=saved_output
        )

        print('第二步：读取Batch范围内的数据')
        batch_input, batch_output, mapper, data = read_file_action(
            start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
            mapper=mapper, data=data, least_words=3, most_word=32, embed_level=embed_level,
            embed_encoder=embed_encoder, bert_dim=bert_dim, binary_label=binary_label,
            path_post_trained=path_post_trained
        )
        sym_ama_list = [sym_ama for _ in batch_input]

        batch_start_index = batch_count
        batch_count += len(batch_input)

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
            lock.acquire()
            all_adj += sym_ama_list.copy()
            all_input += batch_input.copy()
            all_output += batch_output.copy()
            lock.release()
        start_index = mapper['last_index'] + 1
        if batch_count >= stop_after:
            flag = False
        time.sleep(1)
        print('数据已采集：', batch_count, '/', stop_after)
    return all_input, all_adj, all_output


def read_data_new(
        vocab_size=128, limit_text=126, limit_author=2,
        start_index=0,
        data='../my_personality.csv',
        stop_after=None,
        read_file_action=read_file,
        embed_level='word', embed_encoder=encoder_onehot,
        save_by_batch=None,
        bert_dim=8,
        binary_label=False,
        path_post_trained='../../Model/BertDNN/Bert.h5',
        batch_count=0,
        saved_output=None
):
    def has_alive(ts):
        print('===============')
        yes = False
        for thread in ts:
            if thread.is_alive():
                print('线程:', thread.user_id, '进行中')
                yes = True
        print('===============')
        return yes

    def clean_dead(ts):
        i = 0
        while i < len(ts):
            if not ts[i].is_alive():
                ts.pop(i)
            else:
                i += 1
        return ts

    _ = stop_after
    t_lock = threading.Lock()
    mapper_list = []
    adj_mat_list = []
    status = []
    rgl_thread = threading.Thread(
        target=read_graph_loop,
        args=(
            adj_mat_list, bert_dim, data, limit_author, limit_text, mapper_list, path_post_trained, saved_output,
            start_index, vocab_size, t_lock, status
        )
    )
    rgl_thread.start()
    while len(status) <= 0:
        time.sleep(0.1)
    batch_start_index = 0
    break_after_empty = False
    threads = []
    while True:
        time.sleep(0.1)
        if status[0] == 'RGL_FINISH':
            break_after_empty = True
        if len(mapper_list) > 0:
            threads = clean_dead(threads)
            t_lock.acquire()
            mapper = mapper_list.pop(0)
            adj_mat = adj_mat_list.pop(0)
            t_lock.release()
            sd_thread = threading.Thread(
                target=save_data,
                args=(
                    adj_mat, batch_start_index, bert_dim, binary_label, data, embed_encoder, embed_level, limit_author,
                    limit_text, mapper, path_post_trained, read_file_action, save_by_batch, start_index, vocab_size
                )
            )
            setattr(sd_thread, 'user_id', batch_start_index)
            threads.append(sd_thread)
            sd_thread.start()
            start_index = mapper['last_index'] + 1
            batch_start_index += 1
        elif break_after_empty:
            break
        else:
            continue
    while has_alive(threads):
        time.sleep(10)
    batch_rename(save_by_batch, batch_count)


def save_data(adj_mat, batch_start_index, bert_dim, binary_label, data, embed_encoder, embed_level, limit_author,
              limit_text, mapper, path_post_trained, read_file_action, save_by_batch, start_index, vocab_size):
    print('线程:', batch_start_index, '已启动')
    batch_input, batch_output, _, _ = read_file_action(
        start_index=start_index, vocab_size=vocab_size, limit_text=limit_text, limit_author=limit_author,
        mapper=mapper, data=data, least_words=3, most_word=32, embed_level=embed_level,
        embed_encoder=embed_encoder, bert_dim=bert_dim, binary_label=binary_label,
        path_post_trained=path_post_trained
    )
    print('线程:', batch_start_index, '特征矩阵已生成')
    for i in range(len(batch_input)):
        np.save(
            os.path.join(save_by_batch, 'AdjMat_{}_{}.npy'.format(batch_start_index, i)),
            np.array(adj_mat)
        )
        np.save(
            os.path.join(save_by_batch, 'Input_{}_{}.npy'.format(batch_start_index, i)),
            np.array(batch_input[i])
        )
        np.save(
            os.path.join(save_by_batch, 'Output_{}_{}.npy'.format(batch_start_index, i)),
            np.array(batch_output[i])
        )
    print('线程:', batch_start_index, '训练三要素已保存')
    print('线程:', batch_start_index, '结束任务')


if __name__ == '__main__':
    print('Done')
