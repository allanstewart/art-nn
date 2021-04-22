# art-nn
## Pytorch Neural Net GAN + Classifier code

My wife is writing a book about famous painters, and I found some applications of ML for her project:

* kmeans clustering to find a representative color for each artist
* generate artworks to show the characteristics of each painter, using a GAN
* I also wrote a classifier to test how accurate the discriminator model

All art is scaled to 160x160px for training the GAN and classifier, and there's a script (download.py) to source about ~1000 paintings per artist.

There are definitely better libraries out there for pytorch such as eriklindernoren/PyTorch-GAN and junyanz/pytorch-CycleGAN-and-pix2pix. But if you are doing 160x160px or downloading art, these scripts could be of use.

### How to use
`download.py` FIRST for downloading art and downscaling. Populates source/, scaled/, gray/ folders with jpgs
```
echo 'https://www.wikiart.org/en/leonardo-da-vinci' | python3 download.py
```

`kmeans.py` for clustering rgb values in jpgs
```
python3 kmeans.py scaled/leonardo-da-vinci
```

`train_classifier.py` for training multi-class classifier and printing multi-class accuracy. Follow prompts
```
python3 train_classifier.py
```

`train_gan.py` for training GAN. The GAN consists of a generator and discriminator pair, where the generator tries to generate an "artwork" that fools the discriminator. Follow prompts -- it will output a collage of generated images and model
```
python3 train_gan.py
```

`generate_gan.py` only generates image collages for a model trained with `train_gan.py`. Note the usage
```
python3 generate_gan.py [model_filename] [num_channels] [out_file] [num_images:6]
python3 generate_gan.py picasso.torch 3 picasso1.jpg 1
```

### Results

#### Classifier
For a dataset consisting of the artists {joan-miro, pablo-picasso, rembrandt, salvador-dali, vincent-van-gogh}. The accuracy of a uniform random classifier would be 23%. The accuracy achieved with the Classifier model is over 66% using the grayscale model (which performs slightly better than RGB model in 100 epochs). I think the accuracy would be better on an easier dataset, such as by (1) only using colorful art (there's a lot of black and white sketches which are harder to tell apart or (2) using artists from vastly different styles.

#### GAN
I trained GANs on Picasso, Miro, and Dali. Here's an example for Picasso at 100 epochs
![Picasso generated art](/doc/picasso6.jpg)
The art looks like paintings with the right strokes and colors, but is not yet a coherent artwork. I think that the GAN performance could be improved by filtering the input images to examples with a more uniform style. As it is, the artists are very creative and each input artwork is quite different.
