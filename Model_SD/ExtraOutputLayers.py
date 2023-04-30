import tensorflow as tf

from Config_SD import num_steps
from Config_TF import MAX_SL, WORD_VEC_DIM
from DL.Config_FCN import img_shape
from Model_SD.SpatialTransformer import ResBlock, SpatialTransformer


# input.shape = (None, img_shape[0] / 32, img_shape[1] / 32, 512)
# output.shape = (None, img_shape[0], img_shape[1], rgb_channel)
def build_output_block(st_heads=2):
    # input_layer = tf.keras.Input(shape=(None, None, None))
    input_tensor = tf.keras.Input(shape=(int(img_shape[0] / 32), int(img_shape[1] / 32), 512))
    input_tensor_time = tf.keras.Input(shape=(num_steps,))
    input_tensor_context = tf.keras.Input(shape=(MAX_SL, WORD_VEC_DIM))

    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(input_tensor)

    x = ResBlock(64, WORD_VEC_DIM * st_heads)([x, input_tensor_time])
    x = SpatialTransformer(WORD_VEC_DIM * st_heads, st_heads, WORD_VEC_DIM)([x, input_tensor_context])
    x = ResBlock(WORD_VEC_DIM * st_heads, 64)([x, input_tensor_time])

    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    x = ResBlock(64, WORD_VEC_DIM * st_heads)([x, input_tensor_time])
    x = SpatialTransformer(WORD_VEC_DIM * st_heads, st_heads, WORD_VEC_DIM)([x, input_tensor_context])
    x = ResBlock(WORD_VEC_DIM * st_heads, 64)([x, input_tensor_time])

    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    x = ResBlock(64, WORD_VEC_DIM * st_heads)([x, input_tensor_time])
    x = SpatialTransformer(WORD_VEC_DIM * st_heads, st_heads, WORD_VEC_DIM)([x, input_tensor_context])
    x = ResBlock(WORD_VEC_DIM * st_heads, 64)([x, input_tensor_time])

    x = tf.keras.layers.Conv2D(3, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    new_model = tf.keras.Model(inputs=[input_tensor, input_tensor_time, input_tensor_context], outputs=x)
    return new_model


if __name__ == '__main__':
    output_block = build_output_block()
