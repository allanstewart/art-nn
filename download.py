#!/bin/python3
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
import os
import random
import re
import requests
import time
import sys
from urllib.parse import urlparse

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36'
REQUEST_HEADERS = {'User-Agent': USER_AGENT}
RE_RESOLUTION = re.compile(r'(\d+)x(\d+)px')

def delay_get(u):
    time.sleep(random.random())
    return requests.get(u, headers=REQUEST_HEADERS).text

def delay_dl(u, folder):
    print(u)
    time.sleep(random.random())
    os.makedirs(folder, exist_ok=True)
    url_parse = urlparse(u)
    fname = '{}/{}'.format(folder, os.path.basename(url_parse.path))
    with requests.get(u, stream=True, headers=REQUEST_HEADERS) as r, open(fname, 'wb') as fh:
        for chunk in r.iter_content(4096):
            fh.write(chunk)
    return fname

def scale(original, folder):
    fname = '{}/{}'.format(folder, os.path.basename(original))
    os.makedirs(folder, exist_ok=True)
    with Image.open(original) as image:
        bg = Image.new("RGB", image.size, (255, 255, 255))
        split = image.split()
        bg.paste(image, mask=(split[3] if len(split) == 4 else None))
        bg.resize((160, 160), resample=Image.BICUBIC).save(fname, 'JPEG', quality=90)
    return fname

def grayscale(scaled, folder):
    fname = '{}/{}'.format(folder, os.path.basename(scaled))
    os.makedirs(folder, exist_ok=True)
    with Image.open(scaled) as image:
        ImageOps.grayscale(image).save(fname)
    return fname

def for_image(url, aname=None):
    soup = BeautifulSoup(delay_get(url), 'html.parser')
    res_el = soup.find(class_='max-resolution')
    if not res_el:
        print("No go", url, "nores")
        return

    resolution = res_el.get_text()
    w, h = (0, 0)
    re_match = RE_RESOLUTION.match(resolution)
    if re_match:
        w = int(re_match.group(1))
        h = int(re_match.group(2))
    if not re_match or (w*h) > 2**23 or (w/h) > 2 or (w/h) < 0.5:
        print("No go", url, resolution)
        return

    img = soup.find('div', class_='wiki-layout-artist-image-wrapper').find('img')
    src = img['src'].replace('!Large.jpg', '')
    fname = os.path.basename(src)

    folder = 'src{}'.format('/' + aname if aname else '')
    # Don't redownload; remove this if code changes
    if os.path.isfile('{}/{}'.format(folder, fname)):
        return

    # original
    original = delay_dl(src, folder)

    try:
        folder = 'scaled{}'.format('/' + aname if aname else '')
        scaled = scale(original, folder)

        folder = 'gray{}'.format('/' + aname if aname else '')
        grayscaled = grayscale(scaled, folder)
    except OSError:
        print("OSError", url)
        return
    
def for_artist(url):
    url_parse = urlparse(url)
    aname = os.path.basename(url_parse.path)
    soup = BeautifulSoup(delay_get(url), 'html.parser')
    btn = soup.find('a', class_='btn-view-all')
    all_works = '{}://{}{}/text-list'.format(url_parse.scheme, url_parse.netloc, btn['href'])
    soup = BeautifulSoup(delay_get(all_works), 'html.parser')
    for row in soup.find('ul', class_='painting-list-text').find_all('a'):
        for_image('{}://{}{}'.format(url_parse.scheme, url_parse.netloc, row['href']), aname=aname)
    
for my_url in sys.stdin:
    for_artist(my_url.rstrip())
