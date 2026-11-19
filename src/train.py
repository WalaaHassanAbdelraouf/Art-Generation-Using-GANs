# Train a 64x64 DCGAN on a folder of artwork images 

import argparse
from pathlib import Path

import tensorflow as tf
from PIL import Image
import numpy as np


IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 250
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train the artwork DCGAN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/artworks"),
                        help="Root folder containing artist subfolders and images.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_images(data_dir: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(
            f"No images found in '{data_dir}'. Add images inside artist subfolders."
        )
    return [str(path) for path in paths]


def load_image(path):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=IMAGE_CHANNELS,
                                  expand_animations=False)
    image.set_shape([None, None, IMAGE_CHANNELS])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0
    return image


def make_dataset(data_dir: Path, batch_size: int, seed: int):
    image_paths = find_images(data_dir)
    if len(image_paths) < batch_size:
        raise ValueError(
            f"Found {len(image_paths)} images, but batch size is {batch_size}. "
            "Add more images or reduce --batch-size."
        )

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.shuffle(
        buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True
    )
    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    # drop_remainder keeps real and generated batch sizes identical.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(AUTOTUNE), len(image_paths)


def build_generator():
    model = tf.keras.Sequential(name="generator")
    model.add(tf.keras.layers.Input(shape=(LATENT_DIM,)))
    model.add(tf.keras.layers.Dense(4 * 4 * 256, use_bias=False))
    model.add(tf.keras.layers.Reshape((4, 4, 256)))

    for filters in (256, 256, 256, 256):
        model.add(tf.keras.layers.UpSampling2D())
        model.add(tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(IMAGE_CHANNELS, 3, padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    return model


def build_discriminator():
    model = tf.keras.Sequential(name="discriminator")
    model.add(tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)))

    for filters, stride in ((32, 2), (64, 2), (128, 2), (256, 1), (512, 1)):
        model.add(tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same"))
        if filters != 32:
            model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(negative_slope=0.2))
        model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def save_sample_grid(generator, fixed_noise, output_path: Path, epoch: int):
    generated = generator(fixed_noise, training=False)
    generated = ((generated + 1.0) * 127.5).numpy()
    generated = np.clip(generated, 0, 255).astype(np.uint8)

    columns = 7
    rows = int(np.ceil(len(generated) / columns))
    margin = 4
    canvas = Image.new(
        "RGB",
        (columns * IMAGE_SIZE + (columns + 1) * margin,
         rows * IMAGE_SIZE + (rows + 1) * margin),
        color="black",
    )

    for index, array in enumerate(generated):
        image = Image.fromarray(array)
        x = margin + (index % columns) * (IMAGE_SIZE + margin)
        y = margin + (index // columns) * (IMAGE_SIZE + margin)
        canvas.paste(image, (x, y))

    canvas.save(output_path / f"epoch_{epoch:03d}.png")


@tf.function
def train_step(real_images, generator, discriminator,
               generator_optimizer, discriminator_optimizer, batch_size):
    noise = tf.random.normal([batch_size, LATENT_DIM])

    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_gradients = generator_tape.gradient(
        gen_loss, generator.trainable_variables
    )
    discriminator_gradients = discriminator_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )
    return gen_loss, disc_loss


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive integers.")

    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    dataset, image_count = make_dataset(args.data_dir, args.batch_size, args.seed)
    steps_per_epoch = image_count // args.batch_size
    print(f"Found {image_count} images. Training for {args.epochs} epochs.")

    generator = build_generator()
    discriminator = build_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1.2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4, beta_1=0.5)

    fixed_noise = tf.random.normal([49, LATENT_DIM])

    for epoch in range(1, args.epochs + 1):
        generator_losses = []
        discriminator_losses = []

        for real_images in dataset:
            gen_loss, disc_loss = train_step(
                real_images, generator, discriminator,
                generator_optimizer, discriminator_optimizer, args.batch_size
            )
            generator_losses.append(float(gen_loss))
            discriminator_losses.append(float(disc_loss))

        mean_gen_loss = float(np.mean(generator_losses))
        mean_disc_loss = float(np.mean(discriminator_losses))
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"generator_loss: {mean_gen_loss:.4f} - "
            f"discriminator_loss: {mean_disc_loss:.4f}"
        )
        save_sample_grid(generator, fixed_noise, samples_dir, epoch)

    model_path = args.output_dir / "generator.keras"
    generator.save(model_path)
    print(f"Generator saved to: {model_path.resolve()}")
    print(f"Training samples saved to: {samples_dir.resolve()}")


if __name__ == "__main__":
    main()
