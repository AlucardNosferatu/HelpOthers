import math

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from DL.FCN import full_convolution_net_for_sd
from DL.cfg import img_shape
from Model.Transformer import transformer_encoder_only
from config import TGT_VOC_SIZE, N_LAYERS, UNITS, WORD_VEC_DIM, N_HEADS, DROP, MAX_SL
from eval import sent2vec
from tokenizer import task_conv_chn
from Constant import _ALPHAS_CUMPROD


def get_models():
    text_encoder = transformer_encoder_only(
        seq_length=MAX_SL,
        vocab_size=TGT_VOC_SIZE,
        num_layers=N_LAYERS,
        units=UNITS,
        word_vec_dim=WORD_VEC_DIM,
        num_heads=N_HEADS,
        dropout=DROP
    )

    img_diffuser = full_convolution_net_for_sd(
        rgb_channel=3,
        time_step=256,
        time_encoded_dim=64,
        io_boundary=19,
        resb_channel=64,
        st_heads=8
    )
    return text_encoder, img_diffuser


def timestep_embedding(timesteps, dim=256, max_period=10000):
    half = dim // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(0, half, dtype="float32") / half
    )
    args = np.array(timesteps) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return tf.convert_to_tensor(embedding.reshape(1, -1), dtype=tf.float32)


def get_denoise_img(
        model_id,
        batch_size,
        noise_image,
        timestep,
        context,
        empty_context,
        noise_guidance_scale
):
    timestep = np.array([timestep])
    timestep = timestep_embedding(timestep)
    timestep = np.repeat(timestep, batch_size, axis=0)
    denoise_image = model_id.predict_on_batch(
        [noise_image, timestep, empty_context]
    )
    noise_image = model_id.predict_on_batch(
        [noise_image, timestep, context]
    )
    return denoise_image + noise_guidance_scale * (
            noise_image - denoise_image
    )


def add_noise(x, t, noise=None, rgb_channel=3):
    batch_size, w, h = x.shape[0], x.shape[1], x.shape[2]
    if noise is None:
        noise = tf.random.normal((batch_size, w, h, rgb_channel), dtype=tf.float32)
    sqrt_alpha_prod = _ALPHAS_CUMPROD[t] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - _ALPHAS_CUMPROD[t]) ** 0.5

    return sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise


def get_initial_params(
        timesteps,
        batch_size,
        seed,
        input_image=None,
        input_img_noise_t=None,
        rgb_channel=3
):
    n_h = img_shape[0]
    n_w = img_shape[1]
    alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
    alphas_prev = [1.0] + alphas[:-1]
    if input_image is None:
        latent = tf.random.normal((batch_size, n_h, n_w, rgb_channel), seed=seed)
    else:
        latent = tf.repeat(input_image, batch_size, axis=0)
        latent = add_noise(latent, input_img_noise_t)
    return latent, alphas, alphas_prev


def get_x_prev_and_pred_x0(x, e_t, a_t, a_prev):
    sigma_t = 0
    sqrt_one_minus_at = math.sqrt(1 - a_t)
    pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

    # Direction pointing to x_t
    dir_xt = math.sqrt(1.0 - a_prev - sigma_t ** 2) * e_t
    x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
    return x_prev, pred_x0


def get_prompt_img(
        model_id,
        model_te,
        noise_image,
        noise_image_strength,
        prompt,
        seed,
        num_steps,
        noise_guidance_scale,
        batch_size=1
):
    tok, vocab_size = task_conv_chn(None, None, False, False)
    start_tok, end_tok = [vocab_size], [vocab_size + 1]
    prompt_vec = sent2vec(end_tok, prompt, start_tok, tok)
    prompt_vec = np.repeat(prompt_vec, batch_size, axis=0)
    context = model_te.predict_on_batch(prompt_vec)

    input_image_tensor = None
    if noise_image is not None:
        if type(noise_image) is str:
            noise_image = Image.open(noise_image)
            noise_image = noise_image.resize(size=(model_id.input_shape[0][1], model_id.input_shape[0][2]))

        elif type(noise_image) is np.ndarray:
            noise_image = np.resize(
                noise_image,
                (model_id.input_shape[1], model_id.input_shape[2], model_id.input_shape[3])
            )

        input_image_array = np.array(noise_image, dtype=np.float32)[None, ..., :3]
        input_image_tensor = tf.cast((input_image_array / 255.0) * 2 - 1, tf.float32)

    empty_prompt = ''
    empty_prompt_vec = sent2vec(end_tok, empty_prompt, start_tok, tok)
    empty_prompt_vec = np.repeat(empty_prompt_vec, batch_size, axis=0)
    empty_context = model_te.predict_on_batch(empty_prompt_vec)

    timesteps = np.arange(1, 1000, 1000 // num_steps)
    input_img_noise_t = timesteps[int(len(timesteps) * noise_image_strength)]
    latent, alphas, alphas_prev = get_initial_params(
        timesteps, batch_size, seed, input_image=input_image_tensor, input_img_noise_t=input_img_noise_t
    )

    if noise_image is not None:
        timesteps = timesteps[: int(len(timesteps) * noise_image_strength)]

    progbar = tqdm(list(enumerate(timesteps))[::-1])
    for index, timestep in progbar:
        progbar.set_description(f"{index:3d} {timestep:3d}")
        e_t = get_denoise_img(
            model_id=model_id,
            batch_size=batch_size,
            noise_image=latent,
            timestep=timestep,
            context=context,
            empty_context=empty_context,
            noise_guidance_scale=noise_guidance_scale
        )
        a_t, a_prev = alphas[index], alphas_prev[index]
        latent, pred_x0 = get_x_prev_and_pred_x0(
            latent, e_t, a_t, a_prev
        )
    return latent


if __name__ == '__main__':
    mdl_te, mdl_id = get_models()
    get_prompt_img(
        model_id=mdl_id,
        model_te=mdl_te,
        noise_image='FCN/tf2.0-FCN/DL/data/road/test/image_2/um_000000.png',
        noise_image_strength=0.5,
        prompt='老婆，我去上班了',
        seed=None,
        num_steps=25,
        noise_guidance_scale=7.5,
        batch_size=1
    )
    print('Done')
