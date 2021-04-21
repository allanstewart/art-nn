from .init import weights_init
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Linear,
    LeakyReLU,
    Module,
    Upsample,
    Sequential,
    Sigmoid,
    Tanh,
)

class Generator(Module):
    def __init__(self, latent_size, image_size, num_channels):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.l1 = Linear(latent_size, 128 * (self.image_size // 4) ** 2)
        self.layers = Sequential(
            BatchNorm2d(128),
            Upsample(scale_factor=2),
            Conv2d(128, 128, 3, stride=1, padding=1),
            BatchNorm2d(128, 0.8),
            LeakyReLU(0.2, inplace=True),
            Upsample(scale_factor=2),
            Conv2d(128, 64, 3, stride=1, padding=1),
            BatchNorm2d(64, 0.8),
            LeakyReLU(0.2, inplace=True),
            Conv2d(64, num_channels, 3, stride=1, padding=1),
            Tanh(),
        )
        self.layers.apply(weights_init)

    def forward(self, x):
        out = self.l1(x)
        out = out.view(out.shape[0], 128, self.image_size // 4, self.image_size // 4)
        return self.layers(out)

class Discriminator(Module):
    NUM_FEATURE_MAPS = 8

    def __init__(self, num_channels):
        super(Discriminator, self).__init__()
        self.layers = Sequential(
            # input is (nc) x 160 x 160
            Conv2d(num_channels, self.NUM_FEATURE_MAPS, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 80 x 80
            Conv2d(self.NUM_FEATURE_MAPS, self.NUM_FEATURE_MAPS * 2, 4, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 2),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 40 x 40
            Conv2d(self.NUM_FEATURE_MAPS * 2, self.NUM_FEATURE_MAPS * 4, 4, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 4),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 20 x 20
            Conv2d(self.NUM_FEATURE_MAPS * 4, self.NUM_FEATURE_MAPS * 8, 4, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 8),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 10 x 10
            Conv2d(self.NUM_FEATURE_MAPS * 8, 1, 10, 1, 0, bias=False),
            Sigmoid(),
        )
        self.layers.apply(weights_init)

    def forward(self, x):
        return self.layers(x)

# Multi-classifier
class Classifier(Module):
    NUM_FEATURE_MAPS = 8

    def __init__(self, num_channels, num_classes):
        super(Classifier, self).__init__()
        self.layers = Sequential(
            # input is (nc) x 160 x 160
            Conv2d(num_channels, self.NUM_FEATURE_MAPS, 4, 2, 1, bias=False),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 80 x 80
            Conv2d(self.NUM_FEATURE_MAPS, self.NUM_FEATURE_MAPS * 2, 4, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 2),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 40 x 40
            Conv2d(self.NUM_FEATURE_MAPS * 2, self.NUM_FEATURE_MAPS * 4, 4, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 4),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 20 x 20
            Conv2d(self.NUM_FEATURE_MAPS * 4, self.NUM_FEATURE_MAPS * 8, 4, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 8),
            LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 10 x 10
            Conv2d(self.NUM_FEATURE_MAPS * 8, num_classes, 10, 1, 0, bias=False),
            # No need for softmax -- this is done in post-processing
        )
        self.layers.apply(weights_init)

    def forward(self, x):
        return self.layers(x)
