import os
import pickle
import random
import threading

import keras_nlp
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

file_lock = threading.Lock()
bert_lock = threading.Lock()


def build_processor(
        use_post_trained=False, path_post_trained='Bert.h5', saved_output=None
):
    def check_consistency(sav, mlm):
        key = random.choice(list(sav.keys()))
        res = mlm.preprocessor([key])
        vec = mlm.backbone(res)['pooled_output'].numpy()[0].tolist()
        if vec == sav[key]:
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
            "bert_tiny_en_uncased"
        )
    masked_lm.preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
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
    if load:
        bert_lock.acquire()
        hit = hasattr(masked_lm, 'saved_output') and input_str in masked_lm.saved_output.keys()
        bert_lock.release()
        if hit:
            vec = masked_lm.saved_output[input_str]
            return vec
    bert_lock.acquire()
    res = masked_lm.preprocessor([input_str])
    bert_lock.release()
    bert_lock.acquire()
    res = masked_lm.backbone(res)
    bert_lock.release()
    vec = res['pooled_output'].numpy()[0].tolist()
    if save and hasattr(masked_lm, 'saved_output'):
        bert_lock.acquire()
        masked_lm.saved_output.__setitem__(input_str, vec)
        bert_lock.release()
        file_lock.acquire()
        pickle.dump(masked_lm.saved_output, open(masked_lm.saved_output_path, 'wb'))
        file_lock.release()
    return vec


def tokenize(input_str, processor):
    res = processor.preprocessor(input_str)
    vec = res['token_ids']
    return vec


def detokenize(input_vec, processor):
    batch = len(input_vec.shape) == 2
    text = processor.preprocessor.tokenizer.detokenize(input_vec)
    if not batch:
        text = [text]
    text = [item.numpy().decode('utf-8').split(' [SEP] ')[0].replace('[CLS] ', '') for item in text]
    for sep in ['.', '!', '?', ',']:
        for i in range(len(text)):
            while ' ' + sep in text[i]:
                text: list
                text[i] = text[i].replace(' ' + sep, sep)
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
            "bert_tiny_en_uncased",
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
    processor_ = build_processor(
        use_post_trained=use_post_trained, path_post_trained='Bert.h5',
        saved_output='../../Data/BertGCN/SavedBertEmbedding.pkl'
    )
    old_txt = [
        'I miss Carol a lot. Where is she now?',
        'We should be together.'
    ]
    if not batch_test:
        index = random.choice(list(range(len(old_txt))))
        old_txt = old_txt[index]
    vec_ = tokenize(
        old_txt,
        processor_
    )
    new_txt = detokenize(vec_, processor_)
    if type(old_txt) is not list:
        old_txt = [old_txt]
    old_txt = [item.lower() for item in old_txt]
    if len(old_txt) == 1:
        old_txt = old_txt[0]
    assert old_txt == new_txt
    print('use_post_trained:', use_post_trained, 'batch_test:', batch_test, 'OK')


if __name__ == '__main__':
    # bert_train()
    bert_test(use_post_trained=True, batch_test=True)
    bert_test(use_post_trained=True, batch_test=False)
    bert_test(use_post_trained=False, batch_test=True)
    bert_test(use_post_trained=False, batch_test=False)
