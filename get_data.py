import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def fashion_mnist(for_vae: bool = False) -> (tf.data.Dataset, tf.data.Dataset):
    if not for_vae:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype("float32") / 255.
        x_test = x_test.astype("float32") / 255.
        return tf.constant(x_train), tf.constant(x_test)
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        train_images = preprocess_fashion(x_train)
        test_images = preprocess_fashion(x_test)

        train_size = 60000
        batch_size = 32
        test_size = 10000

        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)
        return train_dataset, test_dataset


def digit_mnist() -> (tf.data.Dataset, tf.data.Dataset):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_digits(train_images)
    test_images = preprocess_digits(test_images)

    train_size = 60000
    batch_size = 32
    test_size = 10000

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)

    return train_dataset, test_dataset


def preprocess_digits(images_in: np.ndarray) -> np.ndarray:
    new_shape = [dim for dim in images_in.shape] + [1]
    images_out = images_in.reshape(new_shape) / 255.
    return np.where(images_out > .5, 1.0, 0.0).astype("float32")


def preprocess_fashion(images_in: np.ndarray) -> np.ndarray:
    new_shape = [dim for dim in images_in.shape] + [1]
    images_out = images_in.reshape(new_shape) / 255.
    return images_out.astype("float32")


def add_noise(images: tf.Tensor) -> tf.Tensor:
    noise_factor = tf.constant(0.2)
    images_noisy = images + noise_factor * tf.random.normal(shape=images.shape)
    images_noisy = tf.clip_by_value(images_noisy, clip_value_min=0., clip_value_max=1.)
    return images_noisy


def normalize(array: np.array) -> np.array:
    return (array - array.min()) / (array.max() - array.min())


def ecg():
    # Download the dataset
    url = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
    dataframe = pd.read_csv(url, header=None)

    raw_data = dataframe.to_numpy()
    labels = raw_data[:, -1]  # The last element contains the labels
    data = raw_data[:, 0:-1]  # The other data points are the electrocardiogram data

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21)
    train_data = normalize(train_data).astype("float32")
    test_data = normalize(test_data).astype("float32")

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]
    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]

    return normal_train_data, anomalous_train_data, normal_test_data, anomalous_test_data
