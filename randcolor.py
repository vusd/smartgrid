import argparse
import sys
import numpy as np
import json
import os
from os.path import isfile, join
import keras
from keras.preprocessing import image
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
import scipy
from skimage import color
import math
import numbers
import time
from tqdm import tqdm
from PIL import Image
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from braceexpand import braceexpand
import glob
import random

def generateRandColor(width, height, outpath):
    stale_color = True

    while stale_color:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        rgbstr = '0x{:02x}{:02x}{:02x}'.format(r,g,b)
        outfile = os.path.join(outpath, "{}.png".format(rgbstr))
        stale_color = os.path.exists(outfile)

    im_array = np.zeros([height, width, 3]).astype(np.uint8)
    im_array[:,:] = [r, g, b]
    im = Image.fromarray(im_array)

    im.save(outfile)

def main():
    parser = argparse.ArgumentParser(description="Make N random color images, save to outdir")
    parser.add_argument('--num-colors', default=100, type=int,
                        help="how many images to generate")
    parser.add_argument('--width', default=10, type=int,
                        help="image width")
    parser.add_argument('--height', default=10, type=int,
                        help="image height")
    parser.add_argument('--output-path', default="outputs/colors/rand100_01", type=str,
                         help='path to where to put output files')
    parser.add_argument('--random-seed', default=1, type=int,
                        help='Use a specific random seed (for repeatability)')
    args = parser.parse_args()

    if args.random_seed:
      print("Setting random seed: ", args.random_seed)
      random.seed(args.random_seed)
      np.random.seed(args.random_seed)
      tf.set_random_seed(args.random_seed)

    # make output directory if needed
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for i in range(args.num_colors):
        generateRandColor(args.width, args.height, args.output_path)

if __name__ == '__main__':
    main()
