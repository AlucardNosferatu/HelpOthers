import os

import cv2
import numpy as np
import tensorflow as tf

from DL.cfg import num_epochs, weight_path, weight_name, learning_rate
from Data import load_prompt_from_txt, generator_train
from Model_TF.Transformer import transformer
from Model_SD.StableDiffusion import get_prompt_img, transformer_encoder
from StableDiffusion import full_convolution_net_for_sd
from config import SET_BS, WGT_PATH, EPOCHS, N_LAYERS, UNITS, WORD_VEC_DIM, N_HEADS, DROP
from tokenizer import do_tokenize, task_conv_chn
from train import prepare_model


def loss_fn(y_true, y_pred):
    loss = tf.math.reduce_mean((y_true - y_pred) ** 2)
    return loss


def get_text_encoder():
    tok, v_size = task_conv_chn(None, None, False, False)
    t_full = transformer(
        vocab_size=v_size + 2,
        num_layers=N_LAYERS,
        units=UNITS,
        word_vec_dim=WORD_VEC_DIM,
        num_heads=N_HEADS,
        dropout=DROP,
        name="transformer"
    )
    t_full.load_weights(WGT_PATH)
    text_encoder = transformer_encoder(t_full)
    return text_encoder


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


def get_img_diffuser():
    img_diffuser = full_convolution_net_for_sd(
        time_encoded_dim=64,
        io_boundary=19,
        resb_channel=64,
        st_heads=8
    )
    if os.path.exists(os.path.join(weight_path, weight_name + '.ckpt.index')):
        img_diffuser.load_weights(os.path.join(weight_path, weight_name + '.ckpt'))
    return img_diffuser


def train_img_diffuser():
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
        run_eagerly=True,
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
    # x_wn_, x_tt_, x_ct_, y_ = load_image_from_files()
    with tf.device('/gpu:0'):
        img_diffuser.fit(
            x=generator_train(),
            steps_per_epoch=100,
            epochs=num_epochs,
            callbacks=[
                checkpoint
            ]
        )


if __name__ == '__main__':
    train_img_diffuser()
    mdl_id = get_img_diffuser()

    # train_text_encoder()
    mdl_te = get_text_encoder()

    result = get_prompt_img(
        model_id=mdl_id,
        model_te=mdl_te,
        noise_image='Data_TF/Image/8.JPEG',
        noise_image_strength=0.1,
        prompt='barding black cape celty_sturluson dress dullahan durarara!! headless highres horse horseback_riding '
               'janemere smoke solo',
        seed=None,
        noise_guidance_scale=30,
        batch_size=1
    )
    cv2.imshow('res', np.array(result)[0, :, :, :])
    cv2.waitKey()
    print('Done')
