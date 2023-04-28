import math
import os.path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from Config import num_steps, rgb_channel
from Constant import ALPHAS_CUMPROD
from DL.FCN import full_convolution_net
from DL.cfg import img_shape
from ExtraOutputLayers import build_output_block
from SpatialTransformer import ResBlock, SpatialTransformer, apply_seq
from Utilities import timestep_tensor, add_noise
from config import MAX_SL, WORD_VEC_DIM
from eval import sent2vec
from tokenizer import task_conv_chn


def transformer_encoder(transformer_full: tf.keras.Model):
    input_layer = transformer_full.get_layer(name='inputs')
    input_tensor = input_layer.output
    enc_padding_mask = transformer_full.get_layer(name='enc_padding_mask')(input_tensor)
    enc_outputs = transformer_full.get_layer(name='encoder')(inputs=[input_tensor, enc_padding_mask])
    return tf.keras.Model(inputs=input_layer.input, outputs=enc_outputs)


def full_convolution_net_for_sd(
        time_step=256,
        time_encoded_dim=64,
        io_boundary=19,
        resb_channel=64,
        st_heads=8
):
    def apply(x_, layer_, emb_=None, context_=None):
        if isinstance(layer_, ResBlock):
            x_ = layer_([x_, emb_])
        elif isinstance(layer_, SpatialTransformer):
            x_ = layer_([x_, context_])
        else:
            x_ = layer_(x_)
        return x_

    fcn = full_convolution_net(rgb_channel)

    t_emb = tf.keras.Input((time_step,))
    time_embed = [
        tf.keras.layers.Dense(time_encoded_dim),
        tf.keras.activations.swish,
        tf.keras.layers.Dense(time_encoded_dim),
    ]
    emb = apply_seq(t_emb, time_embed)

    context = tf.keras.layers.Input((MAX_SL, WORD_VEC_DIM))

    x = fcn.layers[io_boundary - 1].output

    middle_block = [
        ResBlock(resb_channel, WORD_VEC_DIM * st_heads),
        SpatialTransformer(WORD_VEC_DIM * st_heads, st_heads, WORD_VEC_DIM),
        ResBlock(WORD_VEC_DIM * st_heads, fcn.layers[io_boundary].input_shape[-1])
    ]
    for layer in middle_block:
        x = apply(x, layer, emb, context)

    # output_block = fcn.layers[io_boundary:]
    # for layer in output_block:
    #     x = apply(x, layer, emb, context)
    output_block = build_output_block()
    x = output_block(x)
    new_model = tf.keras.Model(inputs=[fcn.input, t_emb, context], outputs=x)
    return new_model


def get_denoise_img(
        model_id,
        batch_size,
        img_with_context,
        timestep,
        context,
        empty_context,
        noise_guidance_scale
):
    timestep = timestep_tensor(batch_size, timestep)

    img_without_context = model_id.predict_on_batch(
        [img_with_context, timestep, empty_context]
    )
    img_with_context = model_id.predict_on_batch(
        [img_with_context, timestep, context]
    )
    return img_without_context + noise_guidance_scale * (
            img_with_context - img_without_context
    )


def get_initial_params(
        timesteps,
        batch_size,
        seed,
        input_image=None,
        input_img_noise_t=None
):
    n_h = img_shape[0]
    n_w = img_shape[1]
    alphas = [ALPHAS_CUMPROD[t] for t in timesteps]
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
        noise_guidance_scale,
        batch_size=1
):
    tok, vocab_size = task_conv_chn(None, None, False, False)
    start_tok, end_tok = [vocab_size], [vocab_size + 1]
    prompt_vec = sent2vec(end_tok, prompt, start_tok, tok, ' ')
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

        input_image_array = np.array(noise_image, dtype=np.uint8)[None, ..., :3]
        # input_image_tensor = tf.cast((input_image_array / 255.0) * 2 - 1, tf.float32)
        input_image_tensor = tf.cast(input_image_array / 255.0, tf.float32)

    ec_path = 'Save/empty_context.npy'
    if os.path.exists(ec_path):
        empty_context = np.load(ec_path)
    else:
        empty_prompt = ''
        empty_prompt_vec = sent2vec(end_tok, empty_prompt, start_tok, tok, ' ')
        empty_prompt_vec = np.repeat(empty_prompt_vec, batch_size, axis=0)
        empty_context = model_te.predict_on_batch(empty_prompt_vec)
        np.save(ec_path, empty_context)

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
            img_with_context=latent,
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
    full_convolution_net_for_sd()
