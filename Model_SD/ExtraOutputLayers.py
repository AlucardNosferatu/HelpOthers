import tensorflow as tf

from DL.Config_FCN import img_shape


# input.shape = (None, img_shape[0] / 32, img_shape[1] / 32, 512)
# output.shape = (None, img_shape[0], img_shape[1], rgb_channel)
def build_output_block():
    # input_layer = tf.keras.Input(shape=(None, None, None))
    input_tensor = tf.keras.Input(shape=(int(img_shape[0] / 32), int(img_shape[1] / 32), 512))
    x = tf.keras.layers.UpSampling2D(size=(4, 4))(input_tensor)
    x = tf.keras.layers.Conv2D(64, (4, 4), padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)
    x = tf.keras.layers.Conv2D(3, (8, 8), padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    new_model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return new_model


if __name__ == '__main__':
    output_block = build_output_block()
