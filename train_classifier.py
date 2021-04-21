import numpy
from PIL import Image
import sys
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import (
    DataLoader,
    random_split,
)

from models.active_models import Classifier
from utils.constants import (
    BATCH_SIZE,
    BETA1,
    NUM_EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
)    
from utils.filters import painting_color_filter
from utils.helpers import (
    multi_acc,
    save_model,
)

num_channels = input("Grayscale (1) or RGB (3)? Enter number: ")
if num_channels == '1':
    root = 'gray/'
    num_channels = 1
    data_transforms = transforms.Compose([
        # Anger! Torchvision loads images to RGB even if they are
        # grayscale, so we have to undo it here!!!
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
elif num_channels == '3':
    root = 'scaled/'
    num_channels = 3
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
else:
    print("Must enter 1 or 3. You entered: {}. Try again!".format(num_channels))
    sys.exit(-1)
output_modelfname = input("File name of trained model [leave blank if not needed]: ")

dataset = dset.ImageFolder(root=root, transform=data_transforms)
label_dist = {}
for d in dataset:
    label_dist[d[1]] = label_dist.get(d[1], 0) + 1
max_label = max(label_dist)
num_labels = max_label + 1
print("Num levels", num_labels, "Label Distribution", label_dist)

classifier = Classifier(num_channels=num_channels, num_classes=num_labels)
train_dset, test_dset = random_split(dataset, [len(dataset) - 200, 200], generator=torch.Generator().manual_seed(42))
print("TRAIN", len(train_dset), "TEST", len(test_dset))
train = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True)
test = DataLoader(test_dset, batch_size=len(test_dset))

loss_fn = CrossEntropyLoss()
optimizer = Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

def evaluate(epoch, test):
    with torch.no_grad():
        for i, data in enumerate(test):
            features, labels = data
            y_pred = classifier(features).flatten(-3, -1)
            acc = multi_acc(y_pred, labels)
            print("EPOCH", epoch, "ACC", acc)

evaluate("PRE", test)
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(train):
        features, labels = data
        optimizer.zero_grad()
        y_pred = classifier(features).flatten(-3, -1)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()

    evaluate(epoch, test)

if output_modelfname:
    save_model(classifier, output_modelfname)
