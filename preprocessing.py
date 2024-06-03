import tensorflow as tf
import matplotlib.pyplot as plt


# load image and parse
def load_and_parse(img_path):
    """This function takes an image path and parse it to two image"""
    img = tf.io.read_file(img_path)

    img = tf.io.decode_jpeg(img)

    img = tf.image.resize(img, [256, 512])

    width = tf.shape(img)[1]

    w = width // 2

    original_image = img[:, :w, :]
    transformed_image = img[:, w:, :]

    # resize img
    original_image = tf.image.resize(original_image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    transformed_image = tf.image.resize(transformed_image, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # normalize
    original_image = (tf.cast(original_image, tf.float32) / 127.5) - 1
    transformed_image = (tf.cast(transformed_image, tf.float32) / 127.5) - 1

    return original_image, transformed_image


# random crop
def random_crop(original, transformed):
    """This method crops the images randomly"""
    stacked = tf.stack([original, transformed], axis=0)

    cropped = tf.image.random_crop(stacked, size=[2, 256, 256, 3])

    return cropped[0], cropped[1]


# random jitter
@tf.function
def random_jitter(original, transformed):
    """This method for jittering the images"""
    original, transformed = random_crop(original, transformed)

    if tf.random.uniform(()) > 0.4:
        original = tf.image.flip_left_right(original)

        transformed = tf.image.flip_left_right(transformed)

    return original, transformed


# concated function
def load_dataset(img_file):
    """This method will be concated in the function"""
    original, transformed = load_and_parse(img_file)

    original, transformed = random_jitter(original, transformed)

    return original, transformed


# prepare the dataset

training_dataset = tf.data.Dataset.list_files("../DATASETS/gd/*.jpg")
training_dataset = training_dataset.map(load_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
training_dataset = training_dataset.shuffle(buffer_size=288)
training_dataset = training_dataset.batch(1)



