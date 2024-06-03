```markdown
# TensorFlow Image Processing Pipeline

This document provides an analysis of a TensorFlow-based pipeline for loading, processing, and augmenting a dataset of images for machine learning tasks.

## Import Statements

```python
import tensorflow as tf
import matplotlib.pyplot as plt
```

- `tensorflow` is imported for building and training machine learning models.
- `matplotlib.pyplot` is imported but not used in the provided code. It is typically used for plotting graphs and images.

## Function: `load_and_parse`

```python
def load_and_parse(img_path):
    """This function takes an image path and parses it into two images."""
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, [256, 512])
    
    width = tf.shape(img)[1]
    w = width // 2

    original_image = img[:, :w, :]
    transformed_image = img[:, w:, :]

    original_image = tf.image.resize(original_image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    transformed_image = tf.image.resize(transformed_image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    original_image = (tf.cast(original_image, tf.float32) / 127.5) - 1
    transformed_image = (tf.cast(transformed_image, tf.float32) / 127.5) - 1

    return original_image, transformed_image
```

- **Purpose**: This function reads an image from a file, decodes it, resizes it, and splits it into two equal parts: `original_image` and `transformed_image`.
- **Normalization**: The images are normalized to a range of [-1, 1].

## Function: `random_crop`

```python
def random_crop(original, transformed):
    """This method crops the images randomly."""
    stacked = tf.stack([original, transformed], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, 256, 256, 3])
    return cropped[0], cropped[1]
```

- **Purpose**: This function stacks two images together, randomly crops them, and returns the cropped versions.

## Function: `random_jitter`

```python
@tf.function
def random_jitter(original, transformed):
    """This method applies random jittering to the images."""
    original, transformed = random_crop(original, transformed)

    if tf.random.uniform(()) > 0.4:
        original = tf.image.flip_left_right(original)
        transformed = tf.image.flip_left_right(transformed)

    return original, transformed
```

- **Purpose**: This function applies random cropping and horizontal flipping to the images for data augmentation.
- **Decorator**: `@tf.function` compiles the function into a TensorFlow graph for performance optimization.

## Function: `load_dataset`

```python
def load_dataset(img_file):
    """This method loads and processes an image file."""
    original, transformed = load_and_parse(img_file)
    original, transformed = random_jitter(original, transformed)
    return original, transformed
```

- **Purpose**: This function loads an image file, parses it, and applies data augmentation.

## Prepare the Dataset

```python
training_dataset = tf.data.Dataset.list_files("../DATASETS/gd/*.jpg")
training_dataset = training_dataset.map(load_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
training_dataset = training_dataset.shuffle(buffer_size=288)
training_dataset = training_dataset.batch(1)
```

- **Purpose**: This block of code creates a TensorFlow dataset pipeline for training.
  - `tf.data.Dataset.list_files` lists all JPEG files in the specified directory.
  - `map` applies the `load_dataset` function to each file, processing images in parallel.
  - `shuffle` shuffles the dataset with a buffer size of 288.
  - `batch` groups the dataset into batches of size 1.

## Summary

This TensorFlow pipeline is designed to load, process, and augment images from a specified directory. The augmentation includes resizing, normalization, random cropping, and horizontal flipping to enhance the diversity of the training dataset.
```
---

Here's an analysis of the Keras code for building a generator and a discriminator model, formatted as a Markdown file.

```markdown
# Keras GAN Models

This document provides an analysis of a Keras-based pipeline for building generator and discriminator models for a GAN (Generative Adversarial Network).

## Import Statements

```python
from keras.layers import *
from keras.models import Sequential, Model
import tensorflow as tf
from keras.utils import plot_model
```

- `keras.layers` imports various neural network layers.
- `keras.models` imports `Sequential` and `Model` classes to define the models.
- `tensorflow` is imported for various TensorFlow utilities.
- `keras.utils` imports `plot_model` for visualizing the model architecture.

## Function: `encode`

```python
def encode(filter_size, kernel_size, batch_norm=True):
    """Encode block"""
    initializer = tf.random_normal_initializer(0., 0.02)
    encoder = Sequential()

    encoder.add(Conv2D(filter_size, kernel_size, padding="same", strides=2, use_bias=False))

    if batch_norm:
        encoder.add(BatchNormalization())

    encoder.add(LeakyReLU())

    return encoder
```

- **Purpose**: This function creates an encoding block consisting of a Conv2D layer followed by optional batch normalization and a LeakyReLU activation.
- **Parameters**:
  - `filter_size`: Number of filters for the Conv2D layer.
  - `kernel_size`: Size of the convolution kernel.
  - `batch_norm`: Boolean to include BatchNormalization.

## Function: `decode`

```python
def decode(filter_size, kernel_size, drop_out=False):
    """Decode block"""
    initializer = tf.random_normal_initializer(0., 0.02)
    decoder = Sequential()

    decoder.add(Conv2DTranspose(filter_size, kernel_size, padding="same", strides=2, use_bias=False, kernel_initializer=initializer))

    if drop_out:
        decoder.add(Dropout(0.5))

    decoder.add(ReLU())

    return decoder
```

- **Purpose**: This function creates a decoding block consisting of a Conv2DTranspose layer followed by optional dropout and a ReLU activation.
- **Parameters**:
  - `filter_size`: Number of filters for the Conv2DTranspose layer.
  - `kernel_size`: Size of the convolution kernel.
  - `drop_out`: Boolean to include Dropout.

## Function: `create_generator`

```python
def create_generator():
    """This method creates the generator"""
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

    last = Conv2DTranspose(filters=3, kernel_size=4, padding="same", use_bias=False, strides=2, kernel_initializer=initializer, activation="tanh")

    x = inputs
    skips = []

    for d in downsample:
        x = d(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(upsample, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = last(x)
    generator = Model(inputs=inputs, outputs=x, name="generator")

    plot_model(model=generator, to_file="model_plots/generator_cgan.png", show_dtype=True, show_trainable=True, show_shapes=True, show_layer_names=True, show_layer_activations=True)

    generator.summary()

    return generator
```

- **Purpose**: This function creates the generator model for the GAN.
- **Architecture**:
  - **Downsample**: Series of encoding blocks.
  - **Upsample**: Series of decoding blocks.
  - **Skip Connections**: Used to concatenate features from the downsampling path to the upsampling path.
  - **Final Layer**: Conv2DTranspose to generate the output image.
- **Visualization**: Uses `plot_model` to save the model architecture as an image.

## Function: `create_discriminator`

```python
def create_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    original = Input(shape=(256, 256, 3), name="original")
    transformed = Input(shape=(256, 256, 3), name="Transformed")

    lay_in = concatenate([original, transformed])

    d1 = encode(64, 4, False)(lay_in)
    d2 = encode(128, 4)(d1)
    d3 = encode(256, 4)(d2)

    zeropad1 = ZeroPadding2D()(d3)
    conv = Conv2D(512, 1, kernel_initializer=initializer, use_bias=False)(zeropad1)
    batchnorm = BatchNormalization()(conv)
    leakyrelu = LeakyReLU()(batchnorm)

    zeropad2 = ZeroPadding2D()(leakyrelu)
    last = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zeropad2)

    discriminator = Model(inputs=[original, transformed], outputs=last)

    plot_model(model=discriminator, to_file="model_plots/discriminator_cgan.png", show_dtype=True, show_trainable=True, show_shapes=True, show_layer_names=True, show_layer_activations=True)

    discriminator.summary()

    return discriminator
```

- **Purpose**: This function creates the discriminator model for the GAN.
- **Architecture**:
  - **Inputs**: Takes two images, original and transformed.
  - **Downsampling**: Series of encoding blocks.
  - **Zero Padding**: Added before convolution layers to maintain dimensions.
  - **Final Layer**: Conv2D to produce a single output indicating real or fake.
- **Visualization**: Uses `plot_model` to save the model architecture as an image.

## Summary

This Keras-based code defines a generator and a discriminator model for a GAN. The generator creates images from noise, while the discriminator evaluates their authenticity. The models are built using encoding and decoding blocks, with additional data augmentation and normalization techniques to improve training performance.
```
---
Here's an analysis of the provided Keras code for defining loss functions and optimizers for a GAN, formatted as a Markdown file.

```markdown
# GAN Loss Functions and Optimizers

This document provides an analysis of the Keras-based implementation of loss functions and optimizers for a GAN (Generative Adversarial Network).

## Import Statements

```python
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import tensorflow as tf
```

- `Adam` optimizer is imported from `keras.optimizers` for training the models.
- `BinaryCrossentropy` is imported from `keras.losses` for calculating the loss.
- `tensorflow` is imported for TensorFlow utilities.

## Loss Function: Binary Cross-Entropy

```python
loss = BinaryCrossentropy(from_logits=True)
```

- **Purpose**: Defines the binary cross-entropy loss function with the `from_logits` parameter set to `True`. This is used for both the generator and discriminator losses.

## Function: `generator_loss`

```python
def generator_loss(d_generated, g_output, target):
    # gan loss
    gan_loss = loss(tf.ones_like(d_generated), d_generated)

    # gen loss
    l1_loss = tf.reduce_mean(tf.abs(target - g_output))

    # total
    g_loss_total = gan_loss + (l1_loss * 100)

    return g_loss_total, gan_loss, l1_loss
```

- **Purpose**: Calculates the total loss for the generator.
- **Parameters**:
  - `d_generated`: Discriminator's output for generated images.
  - `g_output`: Generated images from the generator.
  - `target`: Real target images.
- **Components**:
  - **GAN Loss**: Binary cross-entropy loss between the discriminator's output for generated images and a target of ones (indicating these should be real).
  - **L1 Loss**: Mean absolute error between the generated images and the target images, encouraging the generator to produce images close to the target.
  - **Total Loss**: Sum of GAN loss and L1 loss (weighted by 100).

## Function: `discriminator_loss`

```python
def discriminator_loss(real, fake):
    real_loss = loss(tf.ones_like(real), real)
    fake_loss = loss(tf.zeros_like(fake), fake)

    return real_loss + fake_loss
```

- **Purpose**: Calculates the total loss for the discriminator.
- **Parameters**:
  - `real`: Discriminator's output for real images.
  - `fake`: Discriminator's output for fake (generated) images.
- **Components**:
  - **Real Loss**: Binary cross-entropy loss between the discriminator's output for real images and a target of ones.
  - **Fake Loss**: Binary cross-entropy loss between the discriminator's output for fake images and a target of zeros.
- **Total Loss**: Sum of real loss and fake loss.

## Optimizers

```python
generator_optimizer = Adam(learning_rate=0.0002, beta_2=0.999, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_2=0.999, beta_1=0.5)
```

- **Purpose**: Defines Adam optimizers for both the generator and discriminator with specific hyperparameters.
- **Hyperparameters**:
  - `learning_rate`: Learning rate for the optimizer, set to 0.0002.
  - `beta_2`: Second moment decay rate, set to 0.999.
  - `beta_1`: First moment decay rate, set to 0.5.

## Summary

This Keras-based code defines loss functions and optimizers for training a GAN. The generator's loss combines GAN loss and L1 loss to encourage the production of realistic and accurate images. The discriminator's loss ensures it can distinguish between real and fake images. Adam optimizers with specific hyperparameters are used to train both the generator and discriminator efficiently.
```
---
Here's an analysis of the Keras code for defining and training a conditional GAN (CGAN), formatted as a Markdown file.

```markdown
# Conditional GAN (CGAN) Implementation

This document provides an analysis of the Keras-based implementation of a conditional GAN (CGAN), including model definition, training process, and monitoring.

## Import Statements

```python
from keras.models import Model
from keras.callbacks import Callback
from losses_optimizers import generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss
from create_models import create_generator, create_discriminator
from generate_image import generate_image
from keras.metrics import Mean
import tensorflow as tf
from preprocessing import training_dataset
```

- **Model and Callback**: Imported from `keras.models` and `keras.callbacks`.
- **Losses and Optimizers**: Imported from a custom module `losses_optimizers`.
- **Models Creation**: Imported from a custom module `create_models`.
- **Image Generation**: Imported from a custom module `generate_image`.
- **Metrics**: Imported from `keras.metrics` to track training progress.
- **TensorFlow Utilities**: Imported from `tensorflow`.

## Enable Eager Execution

```python
tf.config.run_functions_eagerly(True)
```

- **Purpose**: Enables eager execution to ensure that TensorFlow operations are executed immediately as they are called within the Python code. This is useful for debugging and understanding the flow of operations.

## Class: `CGAN`

```python
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
```

### Class Components
- **Constructor (`__init__`)**: Initializes the CGAN model with generator and discriminator.
- **Compile Method**: Configures the model for training with optimizers and loss functions.
- **Train Step Method**:
  - **Data**: Receives a batch of data (original and transformed images).
  - **Gradient Tape**: Records operations for automatic differentiation.
  - **Loss Calculation**: Computes generator and discriminator losses.
  - **Gradients**: Calculates and applies gradients for optimization.
  - **Metrics**: Updates training metrics.
  - **Return**: Returns a dictionary of metrics for monitoring.

## Class: `Monitor`

```python
class Monitor(Callback):
    def __init__(self):
        self.den = "den"

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            self.model.generator.save("../completed_models/generator_edges_to.h5")
            self.model.discriminator.save("../completed_models/discriminator_edges_to.h5")
            print("Models Saved")

        generate_image(model=self.model.generator, step=epoch, original=self.model.original, transformed=self.model.transformed)
```

### Class Components
- **Constructor (`__init__`)**: Initializes the callback.
- **On Epoch End Method**: 
  - **Model Saving**: Saves the models every 50 epochs.
  - **Image Generation**: Generates and saves sample images using the generator.

## Create Models and Compile CGAN

```python
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
```

- **Model Creation**: Uses custom functions to create the generator and discriminator models.
- **Optimizer and Loss Functions**: Assigns predefined optimizers and loss functions.
- **Callback**: Initializes the monitoring callback.
- **Compile CGAN**: Compiles the CGAN model with specified loss functions and optimizers.

## Train the CGAN

```python
cgan.fit(training_dataset, batch_size=8, epochs=700, callbacks=[cp])
```

- **Fit Method**: Trains the CGAN model using the training dataset.
- **Parameters**:
  - `training_dataset`: The dataset used for training.
  - `batch_size`: Size of the batches for training, set to 8.
  - `epochs`: Number of epochs for training, set to 700.
  - `callbacks`: List of callbacks, including the monitoring callback.

## Summary

This Keras-based code defines and trains a conditional GAN (CGAN) using custom generator and discriminator models. The CGAN class handles model compilation and training, while the Monitor callback saves models and generates images during training. The training process is configured to run for 700 epochs with a batch size of 8, using specific optimizers and loss functions.
```
