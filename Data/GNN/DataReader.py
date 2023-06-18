import time

import nltk
import numpy as np
import pandas as pd
from autocorrect import Speller
from nltk import WordNetLemmatizer, PorterStemmer, pos_tag
from tqdm import tqdm

from Data.NaiveDNN.DataReader import unify_symbol, extract_parenthesis

pos_map = {
    'VBZ': 'v',
    'NN': 'n'
}
nltk.download('averaged_perceptron_tagger')


def count_total(data='my_personality.csv', limit_text=None, limit_author=None):
    if type(data) is str:
        data = pd.read_csv(data)
    text_count_list = []
    author_count_list = []
    for i in tqdm(range(data.shape[0])):
        row = data.iloc[i, :]
        text = row['STATUS']
        if text not in text_count_list:
            if limit_text is not None and len(text_count_list) >= limit_text:
                pass
            else:
                text_count_list.append(text.lower())
        author = row['#AUTHID']
        if author not in author_count_list:
            if limit_author is not None and len(author_count_list) >= limit_author:
                pass
            else:
                author_count_list.append(author)
    return data, text_count_list, author_count_list


def limit_vocab(text_count_list, vocab_size):
    lemmatizer, stemmer, speller = None, None, None
    all_words = []
    for texts in tqdm(text_count_list):
        texts = unify_symbol(texts)
        texts = extract_parenthesis(texts)
        for text in texts:
            text_slices = text.split('.')
            for text_slice in text_slices:
                if len(text_slice) > 0:
                    text_slice, lemmatizer, stemmer, speller = unify_word_form(text_slice, lemmatizer, stemmer, speller)
                    words = [word for word in text_slice.split(' ') if len(word) > 0]
                    all_words += words
    freq_dist = nltk.FreqDist(samples=all_words)
    rest_words = freq_dist.most_common(vocab_size)
    word2index, index2word = {}, {}
    for index in range(len(rest_words)):
        word, _ = rest_words[index]
        word2index.__setitem__(word, index + 1)
        index2word.__setitem__(index + 1, word)
    return word2index, index2word


def read_file(vocab_size=4096, limit_text=2048, limit_author=128, mapper=None, data='my_personality.csv', least_words=3,
              most_word=30):
    if type(data) is str:
        data = pd.read_csv(data)
    if mapper is None:
        data, mapper = get_mapper(data, limit_author, limit_text, vocab_size)
    print('原始数据和Batch数据已载入')
    lemmatizer = None
    stemmer = None
    speller = None
    all_input = []
    all_output = []
    for index in tqdm(range(data.shape[0])):
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


def get_mapper(data, limit_author, limit_text, vocab_size):
    print('提取Batch文档及作者')
    data, text_count_list, author_count_list = count_total(
        data=data,
        limit_text=limit_text,
        limit_author=limit_author
    )
    time.sleep(1)
    print('Batch文档及作者已提取')
    print('提取Batch词汇')
    word2index, index2word = limit_vocab(text_count_list, vocab_size)
    print('Batch词汇已提取')
    mapper = {
        'w2i': word2index,
        'i2w': index2word,
        'tlist': text_count_list,
        'alist': author_count_list,
        'total_dim': len(author_count_list) + len(text_count_list) + vocab_size
    }
    return data, mapper


def unify_word_form(text, lemmatizer=None, stemmer=None, speller=None):
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stemmer is None:
        stemmer = PorterStemmer()
    if speller is None:
        speller = Speller(lang='en')
    text = speller(text)
    words = text.split(' ')
    words = [word.lower() for word in words]
    word_pos = pos_tag(words)
    for j in range(len(words)):
        word = words[j]
        if word_pos[j][1] in ['VBZ', 'NN']:
            word = lemmatizer.lemmatize(word, pos_map[word_pos[j][1]])
        # word = stemmer.stem(word)
        words[j] = word
    text = ' '.join(words)
    return text, lemmatizer, stemmer, speller


if __name__ == '__main__':
    print('Done')
