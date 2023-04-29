import tensorflow as tf

from DL.Config_FCN import img_shape


# input.shape = (None, img_shape[0] / 32, img_shape[1] / 32, 512)
# output.shape = (None, img_shape[0], img_shape[1], rgb_channel)
def build_output_block():
    # input_layer = tf.keras.Input(shape=(None, None, None))
    input_tensor = tf.keras.Input(shape=(int(img_shape[0] / 32), int(img_shape[1] / 32), 512))

    x = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(input_tensor)

    x = tf.keras.layers.Conv2D(128, (32, 32), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('selu')(x)

    x = tf.keras.layers.Conv2DTranspose(64, (32, 32), (2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('selu')(x)

    # x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    x = tf.keras.layers.Conv2D(32, (16, 16), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('selu')(x)

    x = tf.keras.layers.Conv2DTranspose(16, (16, 16), (2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('selu')(x)

    # x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    x = tf.keras.layers.Conv2D(3, (8, 8), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    new_model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return new_model


if __name__ == '__main__':
    output_block = build_output_block()
