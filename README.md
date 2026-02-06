# GAN Image Generation (CelebA Faces)

Deep Convolutional GAN (DCGAN) built in PyTorch to generate realistic 64×64 human face images from the CelebA dataset. This project demonstrates hands-on experience with generative models, adversarial training, and GPU-accelerated deep learning.

## Project Overview

- Trains a DCGAN on the **CelebA** face dataset to synthesize new celebrity-like faces from random noise.
- Implements separate **Generator** and **Discriminator** networks following the DCGAN paper and PyTorch DCGAN tutorial best practices (strided conv/transpose-conv, batch norm, LeakyReLU/Tanh).
- Uses Apple Silicon GPU acceleration via **MPS** backend for faster training on macOS.

Generated sample grids for epochs 1–5 are saved in `outputs/generated_faces_epoch_*.png`.

## Dataset

- **Dataset:** CelebA (CelebFaces Attributes Dataset).
- **Size:** ~200K aligned, cropped face images at 178×218, resized to 64×64 for this project.
- Local structure (after download and move):

```text
gan-image-generation/
  data/
    img_align_celeba/
      000001.jpg
      000002.jpg
      ...
```

## Model Architecture

**Generator**

- Input: 100-dimensional latent vector z reshaped to 100 × 1 × 1.
- 4 transposed-convolutional blocks with BatchNorm + ReLU, upsampling to 64 × 64.
- Output layer: 3-channel image with `Tanh` activation, normalized to [-1, 1].

**Discriminator**

- 4 convolutional blocks with stride 2, BatchNorm + LeakyReLU.
- Final 1×1 conv + `Sigmoid` to output a real/fake probability.

Both models are implemented in `src/models.py`.

## Training

Main training script: `src/train.py`.

Key hyperparameters:

- Batch size: 128
- Image size: 64×64
- Latent dimension z: 100
- Optimizer: Adam, learning rate 0.0002, betas (0.5, 0.999)
- Loss: Binary Cross-Entropy (standard GAN objective)
- Epochs: 5
- Device: `mps` (Apple GPU) if available, otherwise CPU

Sample images are saved at the end of each epoch:

```text
outputs/
  generated_faces_epoch_1.png
  ...
  generated_faces_epoch_5.png
  generator_final.pth
  discriminator_final.pth
```

### How to Run Training

```bash
# Create environment / install deps
pip install -r requirements.txt

# Start training
python3 -m src.train
```

This will train the DCGAN on CelebA and write generated image grids and model weights into the `outputs/` directory.

## Future Work

- Add a Jupyter notebook to:
  - Load `generator_final.pth`
  - Sample new faces from random noise
  - Visualize training losses
- Experiment with more epochs and deeper architectures (e.g., ResNet-style generator) for higher-quality faces.
