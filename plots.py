import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import imageio
import variational_autoencoder as vae


def generate_and_save_cvae_images(model: vae.CVAE, epoch: int, test_sample: tf.Tensor):
    """Plots a grid of predicted images"""
    # Use a MPL backend that does not cause a window to pop up.
    mpl.use("Agg")

    # Encode input, sample the resulting distribution, and then decode to an image
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)

    fig = plt.figure(figsize=(4, 4))
    n_images = predictions.shape[0]  # The number of images should be square for this to work correctly
    nx = np.sqrt(n_images).astype(int)  # Number of columns of images
    ny = nx  # Number of rows of images
    gs = gridspec.GridSpec(nx, ny, figure=fig)

    for j in range(ny):
        for i in range(nx):
            ax = fig.add_subplot(gs[j, i])
            ax.imshow(predictions[j * ny + i, :, :, 0], cmap="gray")
            ax.axis("off")

    # tight_layout minimizes the overlap between 2 sub-plots
    fp = os.path.join(os.getcwd(), "images", f"image_at_epoch_{epoch:04d}.png")
    plt.savefig(fp)
    plt.show()
    mpl.use("TkAgg")  # Revert to the normal MPL backend that shows plots in new windows


def cvae_gif():
    """Creates a gif the saved images created during "generate_and_save_cvae_images"."""
    image_dir = os.path.join(os.getcwd(), "images")
    anim_filepath = os.path.join(image_dir, "cvae.gif")

    with imageio.get_writer(anim_filepath, mode="I") as writer:
        filenames = glob.glob(os.path.join(image_dir, "image*.png"))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


def plot_cvae_latent_space(model: vae.CVAE, n: int, image_size: int):
    """
    Plots n x n digit images decoded from the latent space.
    Image size assumes the image is square and is asking for the the width or height.  Easy to change int the future.
    """

    # Generate z values of image grid by sampling normal distribution
    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
    image_width = image_size * n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (image_size, image_size))
            image[i * image_size: (i + 1) * image_size, j * image_size: (j + 1) * image_size] = digit.numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="Greys_r")
    plt.axis("Off")
    plt.show()
