import os

import tensorflow as tf

from Config_SD import num_steps, rgb_channel
from DL.Config_FCN import weight_path, weight_name
from DL.FCN import full_convolution_net
from Model_SD.ExtraOutputLayers import build_output_block
from Model_SD.SpatialTransformer import ResBlock, SpatialTransformer, apply_seq
from Config_TF import MAX_SL, WORD_VEC_DIM, N_LAYERS, UNITS, N_HEADS, DROP, WGT_PATH
from Model_TF.Transformer import transformer
from Tokenizer import task_conv_chn


def transformer_encoder(transformer_full: tf.keras.Model):
    input_layer = transformer_full.get_layer(name='inputs')
    input_tensor = input_layer.output
    enc_padding_mask = transformer_full.get_layer(name='enc_padding_mask')(input_tensor)
    enc_outputs = transformer_full.get_layer(name='encoder')(inputs=[input_tensor, enc_padding_mask])
    return tf.keras.Model(inputs=input_layer.input, outputs=enc_outputs)


def full_convolution_net_for_sd(
        io_boundary=19,
        resb_channel=64,
        st_heads=4
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

    t_emb = tf.keras.Input((num_steps,))
    time_embed = [
        tf.keras.layers.Dense(num_steps),
        tf.keras.activations.swish,
        tf.keras.layers.Dense(num_steps),
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
    x = output_block([x, emb, context])
    new_model = tf.keras.Model(inputs=[fcn.input, t_emb, context], outputs=x)
    return new_model


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


def get_img_diffuser():
    img_diffuser = full_convolution_net_for_sd(
        io_boundary=19,
        resb_channel=64,
        st_heads=4
    )
    if os.path.exists(os.path.join(weight_path, weight_name + '.ckpt.index')):
        img_diffuser.load_weights(os.path.join(weight_path, weight_name + '.ckpt'))
    return img_diffuser


if __name__ == '__main__':
    full_convolution_net_for_sd()
