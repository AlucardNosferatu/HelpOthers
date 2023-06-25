import os

import keras_nlp
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


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


def train_mlm(data='../../Data/my_personality.csv'):
    assert 'XLA_FLAGS' in list(os.environ.keys())
    if type(data) is str:
        data = pd.read_csv(data)
    features = []
    for i in tqdm(range(data.shape[0])):
        text = data.iloc[i, :]['STATUS'].lower()
        features.append(text)
    # Pretrained language model.
    if os.path.exists('Bert.h5'):
        masked_lm = tf.keras.models.load_model('Bert.h5')
    else:
        masked_lm = keras_nlp.models.BertMaskedLM.from_preset(
            "bert_tiny_en_uncased",
        )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='Bert.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
    )
    with tf.device('/gpu:0'):
        masked_lm.fit(x=features, batch_size=64, verbose=1, epochs=1000, callbacks=[ckpt])
    print('Done')


if __name__ == '__main__':
    train_mlm()
