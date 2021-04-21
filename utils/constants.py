# 160x160px
IMAGE_SIZE = 160

# Batch size for gradient. Arbitrarily chosen
BATCH_SIZE = 16

# Hyperparameters galore
# A smaller learning rate than DCGan was helpful for not bouncing
# out of the loss function. This is 1/10 the reference value
LEARNING_RATE = 0.00002

# Adam momentum term
BETA1 = 0.5

# How many epochs to train
NUM_EPOCHS = 100

# Number of channels (rgb)
NC = 3

# Size of feature maps in generator
NGF = 8

# Size of feature maps in discriminator
NDF = 8

# Size of latent vector
NZ = 20
