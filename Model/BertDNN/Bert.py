import os
import pickle
import random
import threading

import keras_nlp
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

file_lock = threading.Lock()
bert_lock = threading.Lock()


def embed_right_now(bert_in, mlm):
    bert_lock.acquire()
    result = mlm.preprocessor(bert_in)
    bert_lock.release()
    bert_lock.acquire()
    result = mlm.backbone(result)
    bert_lock.release()
    bert_out = result['pooled_output'].numpy()
    return bert_out


def build_processor(
        use_post_trained=False, path_post_trained='Bert.h5', saved_output=None
):
    def check_consistency(sav, mlm):
        key = random.choice(list(sav.keys()))
        vec = embed_right_now([key], mlm)
        if np.all(vec == sav[key]):
            return True
        else:
            return False

    if use_post_trained:
        print('使用后训练BERT来嵌入文本')
        masked_lm = tf.keras.models.load_model(
            path_post_trained,
            custom_objects={'BertMaskedLM': keras_nlp.models.BertMaskedLM}
        )
    else:
        print('使用预训练BERT来嵌入文本')
        masked_lm = keras_nlp.models.BertMaskedLM.from_preset(
            "bert_base_zh"
        )
    masked_lm.preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_base_zh")
    if saved_output is not None:
        setattr(masked_lm, 'saved_output_path', saved_output)
        if os.path.exists(saved_output):
            saved_output = pickle.load(open(saved_output, 'rb'))
            consist = True
            print('缓存随机校验开始')
            for _ in range(8):
                if check_consistency(saved_output, masked_lm):
                    continue
                else:
                    consist = False
                    break
            if not consist:
                print('缓存随机校验失败，清空缓存')
                saved_output = {}
            else:
                print('缓存随机校验成功')
        else:
            saved_output = {}
        setattr(masked_lm, 'saved_output', saved_output)
    return masked_lm


def embed(input_str, masked_lm, save=True, load=True):
    def save_to_file(mlm, bert_in, bert_out):
        bert_lock.acquire()
        file_lock.acquire()
        mlm.saved_output.__setitem__(bert_in, bert_out)
        pickle.dump(mlm.saved_output, open(mlm.saved_output_path, 'wb'))
        file_lock.release()
        bert_lock.release()

    if type(input_str) is str:
        input_str = [input_str]
    if load:
        vec_list = []
        for child in input_str:
            bert_lock.acquire()
            hit = hasattr(masked_lm, 'saved_output') and child in masked_lm.saved_output.keys()
            bert_lock.release()
            if hit:
                vec = masked_lm.saved_output[child]
            else:
                vec = np.squeeze(embed_right_now([child], masked_lm), axis=0)
                if save and hasattr(masked_lm, 'saved_output'):
                    save_to_file(masked_lm, child, vec)
            vec_list.append(vec)
        vec = np.array(vec_list)
    else:
        vec = embed_right_now(input_str, masked_lm)
    return vec


def tokenize(input_str, processor):
    result = processor.preprocessor(input_str)
    vec = result['token_ids']
    return vec


def detokenize(input_vec, processor):
    batch = len(input_vec.shape) == 2
    text = processor.preprocessor.tokenizer.detokenize(input_vec)
    if not batch:
        text = [text]
    text = [item.numpy().decode('utf-8').split(' [SEP] ')[0].replace('[CLS] ', '') for item in text]
    text = [item.replace(' ', '') for item in text]
    if len(text) == 1:
        text = text[0]
    return text


# Bert训练方式不止一种，这里只用了无标记训练任务MLM，也许可以用分数预测本身来训练？？？
def bert_train(data='../../Data/my_personality.csv'):
    assert 'XLA_FLAGS' in list(os.environ.keys())
    if type(data) is str:
        data = pd.read_csv(data)
    features = []
    for i in tqdm(range(data.shape[0])):
        text = data.iloc[i, :]['STATUS'].lower()
        features.append(text)
    # Pretrained language model.
    if os.path.exists('Bert.h5'):
        masked_lm = tf.keras.models.load_model(
            'Bert.h5',
            custom_objects={'BertMaskedLM': keras_nlp.models.BertMaskedLM}
        )
    else:
        masked_lm = keras_nlp.models.BertMaskedLM.from_preset(
            "bert_base_zh",
        )
    masked_lm.trainable = True
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath='Bert.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
    )
    with tf.device('/gpu:0'):
        masked_lm.fit(x=features, batch_size=64, verbose=1, epochs=1000, callbacks=[ckpt])
    print('Done')


def bert_test(use_post_trained=True, batch_test=True):
    processor = build_processor(
        use_post_trained=use_post_trained, path_post_trained='Bert.h5',
        saved_output='SavedBertEmbedding.pkl'
    )
    old_txt = [
        '我很想念我老婆，她在哪里？',
        '我和她本应在一起的。'
    ]
    if not batch_test:
        index = random.choice(list(range(len(old_txt))))
        old_txt = old_txt[index]
    vec_ = tokenize(
        old_txt,
        processor
    )
    new_txt = detokenize(vec_, processor)
    if type(old_txt) is not list:
        old_txt = [old_txt]
    old_txt = [item.lower() for item in old_txt]
    if len(old_txt) == 1:
        old_txt = old_txt[0]
    assert old_txt == new_txt
    print('use_post_trained:', use_post_trained, 'batch_test:', batch_test, 'OK')


if __name__ == '__main__':
    # bert_train()
    # bert_test(use_post_trained=False, batch_test=True)
    # bert_test(use_post_trained=False, batch_test=False)
    processor_ = build_processor(
        use_post_trained=False, path_post_trained='Bert.h5',
        saved_output='SavedBertEmbedding.pkl'
    )
    res = embed(input_str='今年政企客户的需求很多。', masked_lm=processor_)
    print(res)
