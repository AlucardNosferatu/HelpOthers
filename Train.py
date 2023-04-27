import tensorflow as tf

from Data import load_prompt_from_txt
from config import SET_BS, WGT_PATH, EPOCHS
from tokenizer import do_tokenize, task_conv_chn
from train import prepare_model

if __name__ == '__main__':
    new_tokenizer = True
    increment = False
    increment = increment and not new_tokenizer

    i, o, ids = load_prompt_from_txt('Data/Prompt.txt')

    dataset, vocab_size = do_tokenize(i, o, task_conv_chn, new_tokenizer)
    dataset = dataset.batch(SET_BS)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print('数据集分批+配置预取完成')
    mdl = prepare_model(vocab_size + 2)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        WGT_PATH,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        save_freq='epoch'
    )
    # log_metric = LossHistory()
    # cb_list = [ckpt, log_metric]
    cb_list = [ckpt]
    if increment:
        mdl.load_weights(WGT_PATH)
    mdl.fit(dataset, epochs=EPOCHS, callbacks=cb_list)
