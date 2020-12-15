import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from models import *

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

log_dir = "path/to/logs"

checkpoint_dir = "path/to/checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def visualize_samples(x, y, num_samples):
    random_indexes = np.random.choice(x.shape[0], num_samples)
    fig, ax = plt.subplots(1, num_samples * 2, figsize=(17, 17))
    for index, i in enumerate(range(0, num_samples * 2, 2)):
        ax[i].imshow(x[random_indexes[index]] / 255.0)
        ax[i].axis("off")
        ax[i + 1].imshow(y[random_indexes[index]] / 255.0)
        ax[i + 1].axis("off")
    plt.show()


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Generator input', 'Real image', 'Generator prediction']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_dataset, epochs, test_dataset):
    start = time.time()
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for example_input, example_target in test_dataset.take(1):
            generate_images(generator, example_input, example_target)
        for n, (input_image, target) in train_dataset.enumerate():
            train_step(input_image, target, epoch)
        if (epoch + 1) % SAVE_FREQUENCY == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Total time taken for learning {epochs} epochs: {(time.time() - start) / 60} minutes.')
    checkpoint.save(file_prefix=checkpoint_prefix)


def image_to_tensor_(img):
    arr = tf.convert_to_tensor(np.array(img) / 127.5 - 1)
    data = tf.data.Dataset.from_tensors(arr)
    data = data.batch(1)
    return list(data.take(1))[0]
