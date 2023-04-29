import tensorflow as tf

from DL.Config_FCN import img_shape


# input.shape = (None, img_shape[0] / 32, img_shape[1] / 32, 512)
# output.shape = (None, img_shape[0], img_shape[1], rgb_channel)
def build_output_block():
    # input_layer = tf.keras.Input(shape=(None, None, None))
    input_tensor = tf.keras.Input(shape=(int(img_shape[0] / 32), int(img_shape[1] / 32), 512))
    x = tf.keras.layers.Flatten()(input_tensor)
    x = tf.keras.layers.Dense(int(img_shape[0] / 32) * int(img_shape[0] / 32))(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.Dense(64 * 16 * 16)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.Reshape((16, 16, 64),)(x)
    x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)
    x = tf.keras.layers.Conv2D(64, (8, 8), padding='same')(x)
    x = tf.keras.layers.Activation('tanh')(x)
    x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)
    x = tf.keras.layers.Conv2D(3, (8, 8), padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    new_model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return new_model


if __name__ == '__main__':
    output_block = build_output_block()
