import numpy as np
import tensorflow as tf


class CVAE(tf.keras.Model):
    """
    Convolutional variational autoencoder.

    Helpful discussion here:
    https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
    """

    def __init__(self, latent_dim: int, sample: tf.Tensor):
        super(CVAE, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        # In an oridinary autoencoder the latent space does not necessarily have any regular structure to it that is
        # useful for generating new images.  A variational autoencoder seeks to solve this by regularizing the latent
        # space.

        # "The reason why an input is encoded as a distribution with some variance instead of a single point
        # is that it makes possible to express very naturally the latent space regularisation: the distributions
        # returned by the encoder are enforced to be close to a standard normal distribution."

        # "Without a well defined regularisation term, the model can learn, in order to minimise its reconstruction
        # error, to 'ignore' the fact that distributions are returned and behave almost like classic autoencoders
        # (leading to overfitting)."

        # The encoder takes an input image and outputs a mean and log variance which describes a Gaussian distribution.
        # The input does not have an exact representation in the latent/code space which needs to be fed to the decoder,
        # so we sample the distribution and feed this sampled latent representation (or code) to the decoder.

        # When taking into account the reparameterization z = mean + epsilon * std_dev, this distribution approximates
        # the possible z values the input could have (where epsilon is randomly sampled from a normal distribution).
        # There may be multiple z parameters in the 'z encoding'.
        # Sampling the distribution gets you a z encoding.
        # The encoder describes the conditional distribution of p(z|x) (probability of z given x (input)).

        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=sample.shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_dim + latent_dim),  # No activation
        ])

        # Output is 4 variables: 2 means and 2 logvariances, one for each direction in a 2D Gaussian

        # This decoder takes the sampled z value and reconstructs the input.
        # It seems to also be possible to have the decoder output parameters (mean and logvar) for p(x|z),
        # the probability of x given a z encoding.  Then you would sample from this distribution to get a prediction.
        # It may also be possible that the literature refers to the decoder sampling a distribution internally.
        # A bit confusing really.

        # Sample shape will be reconstructed from reparameterized latent space.  Will start at 1/4 of sample image size.
        x_shape = sample.shape[0] // 4
        y_shape = sample.shape[1] // 4
        n_filters = 32

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=x_shape*y_shape*n_filters, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(x_shape, y_shape, n_filters)),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding="same"),  # No activation
        ])

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        """Samples our prior for z ( p(z) ) which is a Gaussian"""
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid: bool = False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @staticmethod
    def log_normal_pdf(sample: tf.Tensor, mean: tf.Tensor, logvar: tf.Tensor, raxis: int = 1):
        """This function computes the natural logarithm of the Gaussian distribution"""
        log2pi = tf.math.log(2. * np.pi)
        ln_gaussian = -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
        return tf.reduce_sum(ln_gaussian, axis=raxis)

    def compute_loss(self, inputs: tf.Tensor):
        # Encode the input (which is a set of variables/pixels) to a distribution
        mean, logvar = self.encode(inputs)
        # We need to keep the trainable variables connected between the layers (no random sample layer separating them)
        # so we reparameterize to obtain z (which is the encoded representation).  Returns 2 z values.
        z = self.reparameterize(mean, logvar)
        # Reconstruct the input
        x_logit = self.decode(z)
        # Calculate difference between reconstruction and inputs
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=inputs)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        # Compute log of normal distribution
        logpz = self.log_normal_pdf(z, tf.constant(0.), tf.constant(0.))
        # Compute log of distribution output by encoder
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        # Loss has two parts: difference between input image and output image, and difference of encoder-output
        # distribution from a normal distribution.
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def train_step(self, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
