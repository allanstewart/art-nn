from PIL import Image
import numpy
import torch

from .constants import IMAGE_SIZE

def save_images(generated_images_tensor, fname, num_images, num_channels):
    """
    Save images from generator to a file
    generated_images_tensor: from generator(noise). First dim should have size num_images**2
    fname: file name to output (.jpg)
    num_images: will generate num_images**2 images in a square
    num_channels: 1 (grayscale) or 3 (RGB)
    """
    if num_channels not in [1, 3]:
        raise 'Num channels must be 1 or 3'
    mode = "RGB" if num_channels == 3 else "L"
    white = (255, 255, 255) if num_channels == 3 else 255
    with Image.new(mode, (IMAGE_SIZE * num_images, IMAGE_SIZE * num_images), white) as collage:
        for sq_num, image_tensor in enumerate(generated_images_tensor):
            # Change (num_channels, width, height) to (width, height, num_channels)
            image_arr = numpy.moveaxis(image_tensor.detach().numpy(), 0, -1)
            if num_channels == 1:
                image_arr = image_arr.reshape((image_arr.shape[0], image_arr.shape[1]))
            # Convert tanh float (-1.0 to 1.0) to RGB values (0 to255)
            image_arr = (image_arr * 127.5 + 127.5).astype('uint8')
            to_paste = Image.fromarray(image_arr, mode=mode)
            x = IMAGE_SIZE * (sq_num % num_images)
            y = IMAGE_SIZE * (sq_num // num_images)
            collage.paste(to_paste, box=(x, y))
        collage.save(fname, 'JPEG', quality=90)

def save_model(model, fname):
    """
    Save model to file
    """
    torch.save(model.state_dict(), fname)

def multi_acc(y_pred, y_test):
    """
    Computes the accuracy of a multi-class classifier.
    y_pred and y_test should be the same size.
    OK to use raw_outputs as y_pred -- this function will softmax for you
    """
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc.item()
