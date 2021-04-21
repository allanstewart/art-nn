from .init import weights_init
from torch.nn import (
    Module,
    MaxPool2d,
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    Linear,
    ReLU,
    Tanh,
    Sequential,
    Dropout,
)

class OldClassifier(Module):
    """
    This is a standard CNN
        Conv/ReLU -> Max Pool -> Conv -> Max Pool -> Dropout -> MLP
    It performs worse than the discriminator-based model (50% acc vs 70% acc)
    """
    def __init__(self, num_classes, image_size, num_channels):
        super(OldClassifier, self).__init__()
        self.cnn = Sequential(
            Conv2d(num_channels, ndf, kernel_size=5, stride=1, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(ndf, ndf * 2, kernel_size=5, stride=1, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.mlp_with_dropout = Sequential(
            Dropout(0.25),
            Linear(ndf * 2 * (image_size // 4) ** 2, 50),
            Linear(50, num_classes),
        )

    def forward(self, input):
        cnn_out = self.cnn(input)
        pre_linear = cnn_out.view(cnn_out.size(0), -1)
        return self.mlp_with_dropout(pre_linear)

class OldGenerator(Module):
    """
    Generator model using ConvTranspose2d to sequentially
    expand the image by 2x2 each time. However, it has a checkerboarding
    effect which makes it worse than the Upscale based Generator model.
    """
    NUM_FEATURE_MAPS = 8

    def __init__(self, latent_size, image_size=None, num_channels=3):
        super(OldGenerator, self).__init__()
        self.layers = Sequential(
            # input is Z, going into a convolution
            ConvTranspose2d(latent_size, self.NUM_FEATURE_MAPS * 16, 4, 1, 0, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 16),
            ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            ConvTranspose2d(self.NUM_FEATURE_MAPS * 16, self.NUM_FEATURE_MAPS * 8, 5, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 8),
            ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            ConvTranspose2d(self.NUM_FEATURE_MAPS * 8, self.NUM_FEATURE_MAPS * 4, 6, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 4),
            ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            ConvTranspose2d(self.NUM_FEATURE_MAPS * 4, self.NUM_FEATURE_MAPS * 2, 4, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS * 2),
            ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            ConvTranspose2d(self.NUM_FEATURE_MAPS * 2, self.NUM_FEATURE_MAPS, 4, 2, 1, bias=False),
            BatchNorm2d(self.NUM_FEATURE_MAPS),
            ReLU(inplace=True),
            ConvTranspose2d(self.NUM_FEATURE_MAPS, num_channels, 4, 2, 1, bias=False),
            Tanh(),
            # state size. (nc) x 160 x 160
        )
        self.layers.apply(weights_init)

    def forward(self, input):
        return self.layers(input)
