# Art Generator - DCGAN

A Generative Adversarial Network for generating new art.
It uses a Deep Convolutional Generative Adversarial Network (DCGAN) to generate 64×64 RGB artwork.

## Project structure

```text
Art-Generation-Using-GANs/
├── src/
│   ├── train.py       # Data loading, model definitions, and training
│   └── generate.py    # Generate images from a saved Generator
├── data/
│   └── artworks/      
├── outputs/           # Saved model and generated samples (created at runtime)
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Requirements

- Python 3.10–3.12
- TensorFlow-compatible environment
- Artwork images arranged in folders

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Prepare the dataset

Place the artwork images under `data/artworks/`

```text
data/
└── artworks/
```

The training script converts images to RGB, resizes them to 64×64, and normalizes pixel values to [-1, 1]. The original notebook used the Kaggle [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) dataset. Download it separately and check its license/usage terms before redistributing it.

## 3. Train

From the project root:

```bash
python src/train.py --data-dir data/artworks --epochs 250 --batch-size 32
```

For a quick smoke test, start with fewer epochs:

```bash
python src/train.py --data-dir data/artworks --epochs 2 --batch-size 8
```

Training saves:

- `outputs/generator.keras` : trained Generator model
- `outputs/samples/` : one generated image grid per epoch

Training a GAN can take a long time. Use a GPU when available. Generated image quality depends on dataset size, dataset quality, hardware, and training time.

## 4. Generate new artwork

After training:

```bash
python src/generate.py --model outputs/generator.keras --count 10
```

Generated images are saved in `outputs/generated/`. You can also set a repeatable random seed:

```bash
python src/generate.py --count 10 --seed 42
```

## Notes

- The original 64×64 DCGAN approach and separate Generator/Discriminator optimizers are retained.
- The training script uses `drop_remainder=True` so each real-image batch matches the generated batch size.
- The Generator is saved separately, so image generation does not require retraining.

