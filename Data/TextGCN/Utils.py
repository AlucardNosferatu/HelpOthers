import time

import nltk
import pandas as pd
from autocorrect import Speller
from nltk import WordNetLemmatizer, PorterStemmer, pos_tag
from tqdm import tqdm

from Data.BertDNN.DataReader import unify_symbol, extract_parenthesis

pos_map = {
    'VBZ': 'v',
    'NN': 'n'
}


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


def count_total(start_index, data='my_personality.csv', limit_text=None, limit_author=None):
    if type(data) is str:
        data = pd.read_csv(data)
    text_count_list = []
    author_count_list = []
    last_index = 0
    for i in tqdm(range(start_index, data.shape[0])):
        row = data.iloc[i, :]
        text = row['STATUS']
        if text not in text_count_list:
            if limit_text is not None and len(text_count_list) >= limit_text:
                pass
            else:
                text_count_list.append(text.lower())
                last_index = i
        author = row['#AUTHID']
        if author not in author_count_list:
            if limit_author is not None and len(author_count_list) >= limit_author:
                pass
            else:
                author_count_list.append(author)
    return data, text_count_list, author_count_list, last_index


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


def get_mapper(start_index, data, limit_author, limit_text, vocab_size, bert_dim=8):
    print('提取Batch文档及作者')
    data, text_count_list, author_count_list, last_index = count_total(start_index, data=data, limit_text=limit_text,
                                                                       limit_author=limit_author)
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
        'total_dim': len(author_count_list) + len(text_count_list) + vocab_size,
        'last_index': last_index,
        'bert_dim': bert_dim
    }
    return data, mapper
