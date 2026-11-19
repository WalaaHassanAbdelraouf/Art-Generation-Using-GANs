# Generate artwork using a previously trained Generator

import argparse
from pathlib import Path

import tensorflow as tf
from PIL import Image
import numpy as np


LATENT_DIM = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Generate artwork with a trained DCGAN.")
    parser.add_argument("--model", type=Path, default=Path("outputs/generator.keras"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generated"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Train the model first with src/train.py."
        )
    if args.count < 1:
        raise ValueError("--count must be a positive integer.")

    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = tf.keras.models.load_model(args.model)

    noise = tf.random.normal([args.count, LATENT_DIM])
    generated_images = generator(noise, training=False)
    generated_images = ((generated_images + 1.0) * 127.5).numpy()
    generated_images = np.clip(generated_images, 0, 255).astype(np.uint8)

    for index, image_array in enumerate(generated_images, start=1):
        image = Image.fromarray(image_array)
        image.save(args.output_dir / f"artwork_{index:03d}.png")

    print(f"Generated {args.count} images in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
