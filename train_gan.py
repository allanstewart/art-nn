"""
Train a GAN for 160x160 images (grayscale or RGB)
At the end, produces a 6x6 square of generated images
and optionally can save generator model to a file
(to use with generate_gan.py later).

No parameters, but it will ask you about options when you execute.
"""
import numpy
from PIL import Image
import torch
import sys

from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

from models.active_models import (
    Discriminator,
    Generator,
)
from utils.constants import (
    BATCH_SIZE,
    BETA1,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NZ,
)
from utils.filters import (
    artist_filter,
    compose_filter,
)
from utils.helpers import (
    save_images,
    save_model,
)

REAL_LABEL = 1.
FAKE_LABEL = 0.

# torchvision ToTensor() converts images to values [0.0, 1.0]
# Whereas the generator generates values using tanh to [-1.0, 1.0]
# This converts [0.0, 1.0] => [-1.0, 1.0]
sigmoidy_to_tanhy = lambda x: 2.0 * x - 1.0

artist = input("Which artist?: ")
num_channels = input("Grayscale (1) or RGB (3)? Enter number: ")
if num_channels == '1':
    root = 'gray/'
    num_channels = 1
    data_transforms = transforms.Compose([
        # Anger! Torchvision loads images to RGB even if they are
        # grayscale, so we have to undo it here!!!
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(sigmoidy_to_tanhy),
    ])
    file_filter = lambda x: artist_filter(artist, x)
elif num_channels == '3':
    root = 'scaled/'
    num_channels = 3
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(sigmoidy_to_tanhy),
    ])
    file_filter = lambda x: compose_filter(artist, x)
else:
    print("Must enter 1 or 3. You entered: {}. Try again!".format(num_channels))
    sys.exit(-1)
output_imagefname = input("File name of generated image (.jpg) [leave blank if not needed]: ")
output_modelfname = input("File name of trained model [leave blank if not needed]: ")

print("LET'S GOOOOOOO!")
dataset = dset.ImageFolder(
    root=root,
    transform=data_transforms,
    is_valid_file=file_filter,
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

generator = Generator(latent_size=NZ, image_size=IMAGE_SIZE, num_channels=num_channels)
discriminator = Discriminator(num_channels=num_channels)

loss_fn = BCELoss()
optimizer_g = Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_d = Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

print("Generator")
print(generator.layers)
print("Discriminator")
print(discriminator.layers)

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(dataloader):
        # Generate labels/noise
        real_images = data[0]
        real_labels = torch.full((real_images.size(0),), REAL_LABEL, dtype=torch.float)
        fake_labels = torch.full((real_images.size(0),), FAKE_LABEL, dtype=torch.float)
        noise = torch.randn(real_images.size(0), NZ)

        # Update generator
        generator.zero_grad()
        fake = generator(noise)
        output = discriminator(fake).view(-1)
        loss_g = loss_fn(output, real_labels)
        loss_g.backward()
        generator_pred = output.mean().item()
        optimizer_g.step()

        # Update discriminator
        discriminator.zero_grad()

        # Train set 1: real images (label=1)
        output = discriminator(real_images).view(-1)
        discriminator_pred_t = output.mean().item()
        loss_t = loss_fn(output, real_labels)
        # Train set 2: fake images from generator (label=0)
        output = discriminator(fake.detach()).view(-1)
        loss_f = loss_fn(output, fake_labels)
        loss_d = (loss_t + loss_f) / 2
        # Use sum of gradients
        loss_d.backward()
        optimizer_d.step()

        if i % 5 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f' % (epoch + 1, NUM_EPOCHS, i + 1, len(dataloader), loss_t.item(), loss_f.item(), discriminator_pred_t, generator_pred))

if output_imagefname:
    num_images = 6
    noise = torch.randn(num_images * num_images, NZ)
    fake_images = generator(noise)
    save_images(fake_images, output_imagefname, num_images, num_channels)

if output_modelfname:
    save_model(generator, output_modelfname)
