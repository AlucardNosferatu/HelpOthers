import tensorflow as tf

from DL.Config_FCN import img_shape


# input.shape = (None, img_shape[0] / 32, img_shape[1] / 32, 512)
# output.shape = (None, img_shape[0], img_shape[1], rgb_channel)
def build_output_block():
    # input_layer = tf.keras.Input(shape=(None, None, None))
    input_tensor = tf.keras.Input(shape=(int(img_shape[0] / 32), int(img_shape[1] / 32), 512))

    # conv2dt_1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(8, 8), strides=(2, 2), padding='same')
    # lrelu_1 = tf.keras.layers.LeakyReLU()
    # x = conv2dt_1(input_tensor)
    # x = lrelu_1(x)
    # bn_1 = tf.keras.layers.BatchNormalization()
    # x = bn_1(x)

    us_1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
    x = us_1(input_tensor)

    conv2dt_2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(8, 8), strides=(2, 2), padding='same')
    lrelu_2 = tf.keras.layers.LeakyReLU()
    x = conv2dt_2(x)
    x = lrelu_2(x)
    bn_2 = tf.keras.layers.BatchNormalization()
    x = bn_2(x)

    # conv2dt_3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(16, 16), strides=(2, 2), padding='same', )
    # lrelu_3 = tf.keras.layers.LeakyReLU()
    # x = conv2dt_3(x)
    # x = lrelu_3(x)
    # bn_3 = tf.keras.layers.BatchNormalization()
    # x = bn_3(x)

    us_2 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
    x = us_2(x)

    conv2dt_4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(16, 16), strides=(2, 2), padding='same')
    lrelu_4 = tf.keras.layers.LeakyReLU()
    x = lrelu_4(x)
    x = conv2dt_4(x)
    bn_4 = tf.keras.layers.BatchNormalization()
    x = bn_4(x)

    conv2dt_5 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(32, 32), strides=(2, 2), padding='same')
    lrelu_5 = tf.keras.layers.LeakyReLU()
    x = lrelu_5(x)
    x = conv2dt_5(x)
    bn_5 = tf.keras.layers.BatchNormalization()
    x = bn_5(x)

    # us_3 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
    # x = us_3(x)

    conv2dt_6 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(32, 32), strides=(1, 1), padding='same',
                                                activation=tf.keras.activations.sigmoid)
    x = conv2dt_6(x)
    new_model = tf.keras.Model(inputs=input_tensor, outputs=x)
    return new_model
