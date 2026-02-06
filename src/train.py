import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tqdm import tqdm

from .models import Generator, Discriminator


# --------------------
# Hyperparameters
# --------------------
BATCH_SIZE = 128
IMAGE_SIZE = 64
EPOCHS = 5          # start small; later you can increase to 30–50
LR = 0.0002
BETA1 = 0.5
Z_DIM = 100
FEATURES_G = 64
FEATURES_D = 64
CHANNELS = 3

# --------------------
# Device (GPU / MPS / CPU)
# --------------------
device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Using device: {device}")

# --------------------
# Data: CelebA faces
# Your images live in: data/img_align_celeba/*.jpg
# --------------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * CHANNELS, [0.5] * CHANNELS),
])

# ImageFolder expects: data/<class_name>/*.jpg
# Your folder is data/img_align_celeba, so we treat it as one class.
dataset = datasets.ImageFolder(
    root="./data",
    transform=transform,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# --------------------
# Models, loss, optimizers
# --------------------
netG = Generator(Z_DIM, CHANNELS, FEATURES_G).to(device)
netD = Discriminator(CHANNELS, FEATURES_D).to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=device)
os.makedirs("outputs", exist_ok=True)

G_losses, D_losses = [], []

print("Starting training...")


# --------------------
# Training loop
# --------------------
for epoch in range(EPOCHS):
    for i, (real_images, _) in enumerate(
        tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    ):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Labels
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # ---- 1. Train Discriminator ----
        netD.zero_grad()

        # Real images
        out_real = netD(real_images)
        lossD_real = criterion(out_real, real_labels)

        # Fake images
        noise = torch.randn(batch_size, Z_DIM, 1, 1, device=device)
        fake_images = netG(noise)
        out_fake = netD(fake_images.detach())
        lossD_fake = criterion(out_fake, fake_labels)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # ---- 2. Train Generator ----
        netG.zero_grad()
        out_fake_for_G = netD(fake_images)
        lossG = criterion(out_fake_for_G, real_labels)
        lossG.backward()
        optimizerG.step()

        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

    # Save sample images each epoch
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    vutils.save_image(
        fake,
        f"./outputs/generated_faces_epoch_{epoch + 1}.png",
        normalize=True,
        nrow=8,
    )
    print(f"Saved sample images for epoch {epoch + 1}")

# Save final models
torch.save(netG.state_dict(), "./outputs/generator_final.pth")
torch.save(netD.state_dict(), "./outputs/discriminator_final.pth")
print("Training completed!")

