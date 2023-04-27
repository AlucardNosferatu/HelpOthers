import tensorflow as tf

from Data import load_prompt_from_txt
from Model.Transformer import transformer
from Models.StableDiffusion import get_models, get_prompt_img
from config import SET_BS, WGT_PATH, EPOCHS, N_LAYERS, UNITS, WORD_VEC_DIM, N_HEADS, DROP
from tokenizer import do_tokenize, task_conv_chn
from train import prepare_model


def train_text_encoder(new_tokenizer=True, increment=False):
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


if __name__ == '__main__':
    # train_text_encoder()
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
    mdl_te, mdl_id = get_models(t_full)
    get_prompt_img(
        model_id=mdl_id,
        model_te=mdl_te,
        noise_image='Data/Image/2.JPEG',
        noise_image_strength=0.5,
        prompt='barding black cape celty_sturluson dress dullahan durarara!! headless highres horse horseback_riding '
               'janemere smoke solo',
        seed=None,
        num_steps=25,
        noise_guidance_scale=7.5,
        batch_size=1
    )
    print('Done')
