import os

import tensorflow as tf

from Config_TF import SET_BS, WGT_PATH, EPOCHS
from DL.Config_FCN import num_epochs, weight_path, weight_name, learning_rate
from Load_SD import load_prompt_from_txt, generator_train
from Metrics_SD import loss_fn
from Model_SD.StableDiffusion import full_convolution_net_for_sd
from Tokenizer import do_tokenize, task_conv_chn
from Train_TF import prepare_model


def train_text_encoder(new_tokenizer=False):
    i, o, ids = load_prompt_from_txt('Data_SD/Prompt.txt')
    dataset, vocab_size = do_tokenize(i, o, task_conv_chn, new_tokenizer)
    dataset = dataset.batch(SET_BS)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print('数据集分批+配置预取完成')
    text_encoder = prepare_model(vocab_size + 2)
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
    if os.path.exists(WGT_PATH + '.index'):
        text_encoder.load_weights(WGT_PATH)
    text_encoder.fit(dataset, epochs=EPOCHS, callbacks=cb_list)


def train_img_diffuser(debug=False):
    img_diffuser = full_convolution_net_for_sd(
        time_encoded_dim=64,
        io_boundary=19,
        resb_channel=64,
        st_heads=8
    )
    if os.path.exists(os.path.join(weight_path, weight_name + '.ckpt.index')):
        img_diffuser.load_weights(os.path.join(weight_path, weight_name + '.ckpt'))
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.00001)
    img_diffuser.compile(
        run_eagerly=debug,
        optimizer=optimizer,
        loss=[
            loss_fn
        ],
        metrics=['accuracy']
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(weight_path, weight_name + '.ckpt'),
        monitor='loss',
        save_weights_only=True,
        verbose=1,
        save_best_only=True,
        save_freq='epoch',
        mode='min'
    )
    with tf.device('/gpu:0'):
        img_diffuser.fit(
            x=generator_train(random_yield=True),
            steps_per_epoch=100,
            epochs=num_epochs,
            callbacks=[
                checkpoint
            ]
        )


if __name__ == '__main__':
    train_img_diffuser(debug=False)
    train_text_encoder(new_tokenizer=False)