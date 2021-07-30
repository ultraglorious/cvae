import tensorflow as tf
import get_data
import plots
import variational_autoencoder as vae


if __name__ == "__main__":
    # MNIST digit dataset
    train_ds, test_ds = get_data.digit_mnist()
    sample = next(iter(train_ds))[0]

    # MNIST fashion dataset
    # train_ds, test_ds = get_data.fashion_mnist(for_vae=True)
    # sample = next(iter(train_ds))[0]

    latent_dim = 2  # set the dimensionality of the latent space to a plane for visualization later
    autoencoder = vae.CVAE(latent_dim=latent_dim, sample=sample)

    epochs = 10
    num_examples_to_generate = 16  # Choose a number of images that has a real square root
    # keeping the random vector constant for generation (prediction) so it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])

    # Make sure num examples is less than batch size if doing this.  Currently that is 32.
    for test_batch in test_ds.take(1):
        test_sample = test_batch[0:num_examples_to_generate, ...]
    test_sample = tf.expand_dims(test_sample, axis=4)
    plots.generate_and_save_cvae_images(autoencoder, 0, test_sample)  # Plot pre-trained images

    for epoch in range(1, epochs + 1):
        start_time = tf.timestamp()
        for train_x in train_ds:
            autoencoder.train_step(train_x)
        end_time = tf.timestamp()

        mean = tf.keras.metrics.Mean()
        for test_x in test_ds:
            mean(autoencoder.compute_loss(test_x))  # update_state method of tf.keras.metrics.Mean()
        elbo = -mean.result()  # Computes the weighted average of the elements added via update_state method
        print(f"Epoch: {epoch}, Test set ELBO: {elbo}, time elapse for current epoch: {end_time - start_time}")

        plots.generate_and_save_cvae_images(autoencoder, epoch, test_sample)
        plots.cvae_gif()
        plots.plot_cvae_latent_space(model=autoencoder, n=10, image_size=sample.shape[0])
