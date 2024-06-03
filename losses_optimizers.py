from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import tensorflow as tf

loss = BinaryCrossentropy(from_logits=True)


# method for generator loss
def generator_loss(d_generated, g_output, target):
    # gan loss
    gan_loss = loss(tf.ones_like(d_generated), d_generated)

    # gen loss
    l1_loss = tf.reduce_mean(tf.abs(target - g_output))

    # total
    g_loss_total = gan_loss + (l1_loss * 100)

    return g_loss_total, gan_loss, l1_loss


# discriminator loss
def discriminator_loss(real, fake):
    real_loss = loss(tf.ones_like(real), real)
    fake_loss = loss(tf.zeros_like(fake), fake)

    return real_loss + fake_loss


# optimizers
generator_optimizer = Adam(learning_rate=0.0002, beta_2=0.999, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_2=0.999, beta_1=0.5)
