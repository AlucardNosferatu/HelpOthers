import tensorflow as tf

from Config_SD import num_steps
from Config_TF import MAX_SL, WORD_VEC_DIM
from DL.Config_FCN import img_shape
from Model_SD.SpatialTransformer import ResBlock, SpatialTransformer


# input.shape = (None, img_shape[0] / 32, img_shape[1] / 32, 512)
# output.shape = (None, img_shape[0], img_shape[1], rgb_channel)
def build_output_block(st_heads=2, resb_channel=64):
    # input_layer = tf.keras.Input(shape=(None, None, None))
    input_tensor = tf.keras.Input(shape=(int(img_shape[0] / 32), int(img_shape[1] / 32), 512))
    input_tensor_time = tf.keras.Input(shape=(num_steps,))
    input_tensor_context = tf.keras.Input(shape=(MAX_SL, WORD_VEC_DIM))

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(input_tensor)

    x = ResBlock(resb_channel, WORD_VEC_DIM * st_heads)([x, input_tensor_time])
    x = SpatialTransformer(WORD_VEC_DIM * st_heads, st_heads, WORD_VEC_DIM)([x, input_tensor_context])
    x = ResBlock(WORD_VEC_DIM * st_heads, resb_channel)([x, input_tensor_time])

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)

    x = ResBlock(resb_channel, WORD_VEC_DIM * st_heads)([x, input_tensor_time])
    x = SpatialTransformer(WORD_VEC_DIM * st_heads, st_heads, WORD_VEC_DIM)([x, input_tensor_context])
    x = ResBlock(WORD_VEC_DIM * st_heads, resb_channel)([x, input_tensor_time])

    x = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)

    x = tf.keras.layers.Conv2D(3, kernel_size=(2, 2), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    new_model = tf.keras.Model(inputs=[input_tensor, input_tensor_time, input_tensor_context], outputs=x)
    return new_model


if __name__ == '__main__':
    output_block = build_output_block()
