from keras.layers import *
from keras.models import Sequential, Model
import tensorflow as tf
from keras.utils import plot_model


# downsample
def encode(filter_size, kernel_size, batch_norm=True):
    """Encode block"""
    initializer = tf.random_normal_initializer(0., 0.02)
    encoder = Sequential()

    encoder.add(Conv2D(filter_size, kernel_size, padding="same", strides=2, use_bias=False))

    # batch norm
    if batch_norm:
        encoder.add(BatchNormalization())

    # activation
    encoder.add(LeakyReLU())

    return encoder


# upsample
def decode(filter_size, kernel_size, drop_out=False):
    """decode block"""
    initializer = tf.random_normal_initializer(0., 0.02)

    decoder = Sequential()

    decoder.add(Conv2DTranspose(filter_size, kernel_size, padding="same", strides=2, use_bias=False,
                                kernel_initializer=initializer))

    if drop_out:
        decoder.add(Dropout(0.5))

    decoder.add(ReLU())

    return decoder


def create_generator():
    """This method create the generator"""
    inputs = Input(shape=(256, 256, 3))
    downsample = [
        encode(64, 4, False),
        encode(128, 4, False),
        encode(256, 4),
        encode(512, 4),
        encode(512, 4),
        encode(1024, 4),
        encode(1024, 4),
        encode(1024, 4),
    ]

    upsample = [
        decode(1024, 4, True),
        decode(1024, 4, True),
        decode(512, 4, True),
        decode(512, 4),
        decode(256, 4),
        decode(128, 4),
        decode(64, 4)
    ]

    output_channel = 3

    initializer = tf.random_normal_initializer(0., 0.02)

    last = Conv2DTranspose(filters=3, kernel_size=4, padding="same", use_bias=False,
                           strides=2, kernel_initializer=initializer, activation="tanh")

    # concat all layers
    # x and skip
    x = inputs
    skips = []

    # downsample concat
    for d in downsample:
        x = d(x)
        skips.append(x)

    # upsample concat
    skips = reversed(skips[:-1])
    for up, skip in zip(upsample, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    # model output
    x = last(x)

    generator = Model(inputs=inputs, outputs=x, name="generator")

    # plot generator
    plot_model(model=generator, to_file="model_plots/generator_cgan.png", show_dtype=True, show_trainable=True,
               show_shapes=True, show_layer_names=True, show_layer_activations=True)

    generator.summary()

    return generator


def create_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    # two inputs concat
    original = Input(shape=(256, 256, 3), name="original")
    transformed = Input(shape=(256, 256, 3), name="Transformed")

    # layer input
    lay_in = concatenate([original, transformed])

    # down sampling
    d1 = encode(64, 4, False)(lay_in)
    d2 = encode(128, 4)(d1)
    d3 = encode(256, 4)(d2)

    # zero pad -> 31x31x512
    zeropad1 = ZeroPadding2D()(d3)
    conv = Conv2D(512, 1, kernel_initializer=initializer, use_bias=False)(zeropad1)
    batchnorm = BatchNormalization()(conv)
    leakyrelu = LeakyReLU()(batchnorm)

    # zeropad 2
    zeropad2 = ZeroPadding2D()(leakyrelu)
    last = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zeropad2)

    discriminator = Model(inputs=[original, transformed], outputs=last)

    plot_model(model=discriminator, to_file="model_plots/discriminator_cgan.png", show_dtype=True, show_trainable=True,
               show_shapes=True, show_layer_names=True, show_layer_activations=True)

    discriminator.summary()

    return discriminator
