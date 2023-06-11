import numpy as np
import pandas as pd
from tqdm import tqdm

from TextEncoder import tokenize, build_processor


def unify_symbol(text):
    text = text.strip()
    text = text.replace(',', '.').replace('!', '.').replace('?', '.')
    text = text.replace('[', '(').replace('{', '(').replace(']', ')').replace('}', ')')
    return text


def extract_parenthesis(text, texts=None):
    if texts is None:
        texts = []
    in_p = False
    inner_text = []
    lp_stack = []
    for char in list(text):
        if char == '(':
            lp_stack.insert(0, '(')
            in_p = True
        elif char == ')':
            if '(' in lp_stack:
                lp_stack.pop(0)
            else:
                break
            if len(lp_stack) == 0:
                inner_text.pop(0)
                it_str = ''.join(inner_text)
                inner_text.clear()
                text = text.replace('(' + it_str + ')', '')
                if '(' in text and ')' in text:
                    texts = extract_parenthesis(text, texts)
                else:
                    texts.append(text)
                return extract_parenthesis(it_str, texts)
        if in_p:
            inner_text.append(char)
    texts.append(text)
    return texts


def read_file(data_file_path='my_personality.csv', processor=None, tokenize_batch=32, least_words=3, most_word=30):
    if processor is None:
        processor = build_processor(seq_len=32)
    data_csv = pd.read_csv(data_file_path)
    all_input = []
    all_output = []
    temp_list = []
    for index in tqdm(range(data_csv.shape[0])):
        row = data_csv.iloc[index, :]
        text = row['STATUS']
        s_ext = row['sEXT']
        s_neu = row['sNEU']
        s_agr = row['sAGR']
        s_con = row['sCON']
        s_opn = row['sOPN']
        score = [s_ext, s_neu, s_agr, s_con, s_opn]
        text = unify_symbol(text)
        texts = extract_parenthesis(text)
        for text in texts:
            text_slices = text.split('.')
            for text_slice in text_slices:
                if least_words < len(text_slice.split(' ')) < most_word:
                    score_vec = np.array(score)
                    all_output.append(score_vec)
                    temp_list.append(text_slice.lower())
                    if len(temp_list) >= tokenize_batch:
                        text_vec = tokenize(temp_list, processor)
                        for i in range(tokenize_batch):
                            all_input.append(text_vec[i, :])
                        temp_list.clear()
                        assert len(all_input) == len(all_output)
    if len(temp_list) > 0:
        text_vec = tokenize(temp_list, processor)
        for i in range(len(temp_list)):
            all_input.append(text_vec[i, :])
        temp_list.clear()
    assert len(all_input) == len(all_output)
    return all_input, all_output


if __name__ == '__main__':
    print('Done')
