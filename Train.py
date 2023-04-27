import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from Config import num_steps, rgb_channel
from DL.FCN import full_convolution_net_for_sd
from DL.cfg import img_shape, num_epochs, weight_path, weight_name, learning_rate, batch_size
from Data import load_prompt_from_txt
from Model.Transformer import transformer
from Models.StableDiffusion import get_prompt_img, assemble_encoder
from Utilities import timestep_tensor, add_noise
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
    text_encoder = assemble_encoder(t_full)
    return text_encoder


def train_text_encoder(new_tokenizer=False):
    i, o, ids = load_prompt_from_txt('Data/Prompt.txt')
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
        rgb_channel=rgb_channel,
        time_step=num_steps,
        time_encoded_dim=64,
        io_boundary=19,
        resb_channel=64,
        st_heads=8
    )
    if os.path.exists(os.path.join(weight_path, weight_name + '.ckpt.index')):
        img_diffuser.load_weights(os.path.join(weight_path, weight_name + '.ckpt'))
    return img_diffuser


def train_img_diffuser():
    def datasets_id():
        context = np.squeeze(np.load('Save/empty_context.npy'))
        img_dir = 'Data/Image'
        img_files = os.listdir(img_dir)
        img_files = [os.path.join(img_dir, file) for file in img_files]
        timesteps = np.arange(1, 1000, 1000 // num_steps)
        x_wn = []
        x_tt = []
        x_ct = []
        y = []
        for img_file in tqdm(img_files):
            image = cv2.resize(cv2.imread(img_file), img_shape)
            image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
            image = np.array(cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB))
            image = image.astype('float32') / 255.0

            timestep = random.choice(timesteps)
            ts_tensor = np.squeeze(timestep_tensor(1, timestep))

            with_noise = np.squeeze(
                add_noise(np.reshape(image, (1, img_shape[0], img_shape[1], rgb_channel)), timestep))

            x_wn.append(with_noise)
            x_tt.append(ts_tensor)
            x_ct.append(context)
            y.append(image)
        x_wn = np.array(x_wn)
        x_tt = np.array(x_tt)
        x_ct = np.array(x_ct)
        y = np.array(y)
        return x_wn, x_tt, x_ct, y

    img_diffuser = full_convolution_net_for_sd(
        rgb_channel=rgb_channel,
        time_step=num_steps,
        time_encoded_dim=64,
        io_boundary=19,
        resb_channel=64,
        st_heads=8
    )
    if os.path.exists(os.path.join(weight_path, weight_name + '.ckpt.index')):
        img_diffuser.load_weights(os.path.join(weight_path, weight_name + '.ckpt'))
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=0.00001)
    img_diffuser.compile(
        # run_eagerly=True,
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
    x_wn_, x_tt_, x_ct_, y_ = datasets_id()
    with tf.device('/gpu:0'):
        img_diffuser.fit(
            x=[x_wn_, x_tt_, x_ct_],
            y=y_,
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=[
                checkpoint
            ]
        )


if __name__ == '__main__':
    train_img_diffuser()
    mdl_id = get_img_diffuser()

    train_text_encoder()
    mdl_te = get_text_encoder()

    get_prompt_img(
        model_id=mdl_id,
        model_te=mdl_te,
        noise_image='Data/Image/2.JPEG',
        noise_image_strength=0.5,
        prompt='barding black cape celty_sturluson dress dullahan durarara!! headless highres horse horseback_riding '
               'janemere smoke solo',
        seed=None,
        noise_guidance_scale=7.5,
        batch_size=1
    )
    print('Done')
