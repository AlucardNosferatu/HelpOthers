import math
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from Config_SD import rgb_channel, num_steps
from DL.Config_FCN import img_shape, batch_size
from Model_SD.Constant import ALPHAS_CUMPROD
from Model_SD.StableDiffusion import get_img_diffuser, get_text_encoder
from Test_TF import sent2vec
from Tokenizer import task_conv_chn
from Utilities_SD import timestep_tensor, add_noise


def get_denoise_img(
        model_id,
        bsize,
        img_with_context,
        timestep,
        context,
        empty_context,
        noise_guidance_scale
):
    timestep = timestep_tensor(bsize, timestep)

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
        bsize,
        seed,
        input_image=None,
        input_img_noise_t=None
):
    n_h = img_shape[0]
    n_w = img_shape[1]
    alphas = [ALPHAS_CUMPROD[t] for t in timesteps]
    alphas_prev = [1.0] + alphas[:-1]
    if input_image is None:
        latent = tf.random.normal((bsize, n_h, n_w, rgb_channel), seed=seed)
    else:
        latent = tf.repeat(input_image, bsize, axis=0)
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
        bsize=1
):
    tok, vocab_size = task_conv_chn(None, None, False, False)
    start_tok, end_tok = [vocab_size], [vocab_size + 1]
    prompt_vec = sent2vec(end_tok, prompt, start_tok, tok, ' ')
    prompt_vec = np.repeat(prompt_vec, bsize, axis=0)
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

    empty_context = get_empty_context(model_te, start_tok, end_tok, tok)

    timesteps = np.arange(1, 1000, 1000 // num_steps)
    input_img_noise_t = timesteps[int(len(timesteps) * noise_image_strength)]
    latent, alphas, alphas_prev = get_initial_params(
        timesteps, bsize, seed, input_image=input_image_tensor, input_img_noise_t=input_img_noise_t
    )

    if noise_image is not None:
        timesteps = timesteps[: int(len(timesteps) * noise_image_strength)]

    progbar = tqdm(list(enumerate(timesteps))[::-1])
    for index, timestep in progbar:
        progbar.set_description(f"{index:3d} {timestep:3d}")
        e_t = get_denoise_img(
            model_id=model_id,
            bsize=bsize,
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


def get_empty_context(model_te, start_tok=None, end_tok=None, tok=None, force_update=False):
    if tok is None:
        tok, vocab_size = task_conv_chn(None, None, False, False)
    vocab_size = len(tok[0]) + 1
    if start_tok is None:
        start_tok = [vocab_size]
    if end_tok is None:
        end_tok = [vocab_size + 1]

    ec_path = 'Save_SD/empty_context.npy'
    if os.path.exists(ec_path) and not force_update:
        empty_context = np.load(ec_path)
    else:
        empty_prompt = ''
        empty_prompt_vec = sent2vec(end_tok, empty_prompt, start_tok, tok, ' ')
        empty_prompt_vec = np.repeat(empty_prompt_vec, 1, axis=0)
        empty_context = model_te.predict_on_batch(empty_prompt_vec)
        np.save(ec_path, empty_context)
    return empty_context


if __name__ == '__main__':
    mdl_id = get_img_diffuser()
    mdl_te = get_text_encoder()
    get_empty_context(model_te=mdl_te)
    result = get_prompt_img(
        model_id=mdl_id,
        model_te=mdl_te,
        noise_image='Data_SD/Image/8.JPEG',
        noise_image_strength=0.1,
        prompt='barding black cape celty_sturluson dress dullahan durarara!! headless highres horse horseback_riding '
               'janemere smoke solo',
        seed=None,
        noise_guidance_scale=30,
        bsize=1
    )
    cv2.imshow('res', np.array(result)[0, :, :, :])
    cv2.waitKey()
    print('Done')
