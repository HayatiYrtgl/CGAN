from keras.models import Model
from keras.callbacks import Callback
from losses_optimizers import generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss
from create_models import create_generator, create_discriminator
from generate_image import generate_image
from keras.metrics import Mean
import tensorflow as tf
from preprocessing import training_dataset
tf.config.run_functions_eagerly(True)
# model
class CGAN(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.original = None
        self.transformed = None

    def compile(self, g_opt, d_opt, g_loss, d_loss):
        super().compile()
        self.generator_optimizer = g_opt
        self.discriminator_optimizer = d_opt
        self.generator_loss = g_loss
        self.discriminator_loss = d_loss
        self.gan_metrics = Mean(name="GAN_loss")
        self.gen_metrics = Mean(name="generator_loss")
        self.l1_metrics = Mean(name="l1_loss")
        self.disc_metrics = Mean(name="disc_metrics")

    def train_step(self, data):
        original, transformed = data

        self.original, self.transformed = original, transformed

        # gradient
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # generated_image
            generated_image = self.generator(original, training=True)

            # discriminate
            disc_real_output = self.discriminator([original, transformed], training=True)
            disc_fake_output = self.discriminator([original, generated_image], training=True)

            gen_total_loss, gan_loss, l1_loss = self.generator_loss(disc_fake_output, generated_image, transformed)
            disc_loss = self.discriminator_loss(disc_real_output, disc_fake_output)

        # gradients
        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_weights)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)

        # optimize
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_weights))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_weights))

        self.gen_metrics.update_state(gen_total_loss)
        self.gan_metrics.update_state(gan_loss)
        self.l1_metrics.update_state(l1_loss)
        self.disc_metrics.update_state(disc_loss)

        # generate image
        return {
            "Gan loss :": self.gan_metrics.result(),
            "Generator_loss: ": self.gen_metrics.result(),
            "L1_loss :": self.l1_metrics.result(),
            "Disc loss :": self.disc_metrics.result()
        }


class Monitor(Callback):
    def __init__(self):
        self.den = "den"

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            self.model.generator.save("../completed_models/generator_edges_to.h5")
            self.model.discriminator.save("../completed_models/discriminator_edges_to.h5")
            print("Models Saved")

        generate_image(model=self.model.generator, step=epoch, original=self.model.original, transformed=self.model.transformed)


# create variables
generator = create_generator()
discriminator = create_discriminator()
generator_optimizer = generator_optimizer
discriminator_optimizer = discriminator_optimizer
generator_loss = generator_loss
discriminator_loss = discriminator_loss
training_dataset = training_dataset

cp = Monitor()

cgan = CGAN(generator=generator, discriminator=discriminator)
cgan.compile(g_loss=generator_loss, g_opt=generator_optimizer, d_loss=discriminator_loss, d_opt=discriminator_optimizer)
cgan.fit(training_dataset, batch_size=8, epochs=700, callbacks=[cp])