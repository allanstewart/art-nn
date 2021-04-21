import numpy
from PIL import Image

def artist_filter(artist, path):
    """
    Filters one specific artist (as opposed to all images)
    """
    return (artist + '/') in path

def painting_color_filter(path):
    """
    Checks if a painting has interesting colors.
    If the painting were grayscale, then the colors would be (x,x,x)
    and therefore amax/sum would be 1/3.
    Filtering above 0.35 filters out pure grayscale art.
    """
    with Image.open(path) as image:
        arr = numpy.array(image)
        sat = numpy.amax(arr, axis=-1) / numpy.sum(arr, axis=-1)
        return sat.mean() > 0.35

def compose_filter(artist, path):
    """
    Composes the above two filters
    """
    return artist_filter(artist, path) and painting_color_filter(path)
