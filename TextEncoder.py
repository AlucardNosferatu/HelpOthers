import keras_nlp
import numpy as np


def build_processor(seq_len=32):
    tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_en_uncased")
    processor = keras_nlp.models.BertPreprocessor(tokenizer=tokenizer, sequence_length=seq_len)
    return processor


def tokenize(input_str, processor):
    vec = np.array(processor(input_str)['token_ids'])
    return vec


def detokenize(input_vec, processor):
    txt = np.array(processor.tokenizer.detokenize(input_vec))
    return txt
