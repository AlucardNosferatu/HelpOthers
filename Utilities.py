import math

import numpy as np
import tensorflow as tf

from Config import num_steps
from Constant import ALPHAS_CUMPROD


def timestep_tensor(batch_size, timestep):
    timestep = np.array([timestep])
    timestep = timestep_embedding(timestep)
    timestep = np.repeat(timestep, batch_size, axis=0)
    return timestep


def timestep_embedding(timesteps, max_period=10000):
    half = num_steps // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(0, half, dtype="float32") / half
    )
    args = np.array(timesteps) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return tf.convert_to_tensor(embedding.reshape(1, -1), dtype=tf.float32)


def add_noise(x, t, noise=None, rgb_channel=3):
    batch_size, w, h = x.shape[0], x.shape[1], x.shape[2]
    if noise is None:
        noise = tf.random.normal((batch_size, w, h, rgb_channel), mean=0, dtype=tf.float32)
        min_n = np.min(np.array(noise))
        noise = noise - min_n
        max_n = np.max(np.array(noise))
        noise = noise / max_n
        noise = noise - 0.5
        noise = noise / 4
    sqrt_alpha_prod = ALPHAS_CUMPROD[t] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - ALPHAS_CUMPROD[t]) ** 0.5
    res = sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise
    res = tf.clip_by_value(res, clip_value_min=0.0, clip_value_max=1.0)
    return res
