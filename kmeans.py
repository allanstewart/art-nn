"""
K-means clusters pixel colors for images.
Output is 10 clusters in RGB space of "representative" colors
"""
from PIL import Image
import glob
import numpy
from sklearn.cluster import KMeans
import sys
import zlib

artist = sys.argv[1]
def get_pixels(artist):
    globby = '{}/*.jpg'.format(artist)

    arrs = []
    for f_name in glob.glob(globby):
        image = Image.open(f_name)
        numpy_image = numpy.array(image)
        shape = numpy_image.shape
        if len(shape) == 2:
            continue
        if len(shape) != 3:
            raise 'shape wrong'
        if shape[2] == 4:
            numpy_image = numpy_image[:, :, 0:3]
        elif shape[2] != 3:
            raise 'shape wrong'
        arrs.append(numpy_image.reshape(shape[0] * shape[1], 3))
    return numpy.concatenate(tuple(arrs))

pixels = get_pixels(artist)
seed = zlib.crc32(artist.encode('utf-8'))
kmeans = KMeans(
    init="random",
    n_clusters=10,
    n_init=10,
    max_iter=80,
    random_state=seed,
)
kmeans.fit(pixels)
labels = numpy.bincount(kmeans.labels_)
for i, cluster in sorted(enumerate(kmeans.cluster_centers_), key=lambda i: labels[i[0]], reverse=True):
    print("rgb({},{},{})".format(round(cluster[0]), round(cluster[1]), round(cluster[2])), labels[i])
