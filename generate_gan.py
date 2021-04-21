"""
Given a generator model trained with train_gan.py, generate more images.
usage: python3 generate_gan.py [model_filename] [num_channels] [out_file] [num_images:6]
"""
import sys
import torch

from models.active_models import Generator
from utils.constants import (
    IMAGE_SIZE,
    NZ,
)
from utils.helpers import save_images

if len(sys.argv) < 4:
    print("usage: python3 generate_gan.py [model_filename] [num_channels] [out_file] [num_images:6]")
    sys.exit(-1)
model_fname = sys.argv[1]
num_channels = sys.argv[2]
if num_channels == '1':
    num_channels = 1
elif num_channels == '3':
    num_channels = 3
else:
    print("num_channels: Must enter 1 or 3. You entered: {}. Try again!".format(num_channels))
output_imagefname = sys.argv[3]
num_images = int(sys.argv[4] if len(sys.argv) > 4 else 6)

generator = Generator(latent_size=NZ, image_size=IMAGE_SIZE, num_channels=num_channels)
generator.load_state_dict(torch.load(model_fname))
generator.eval()

noise = torch.randn(num_images * num_images, NZ)
fake_images = generator(noise)
save_images(fake_images, output_imagefname, num_images, num_channels)
