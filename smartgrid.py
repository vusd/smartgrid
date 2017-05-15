import argparse
import sys
import numpy as np
import json
import os
from os.path import isfile, join
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
import scipy
import math
import numbers

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from braceexpand import braceexpand
import glob

# import with fallback behavior
using_lapjv1 = False
try:
    # https://github.com/src-d/lapjv
    import lapjv
except ImportError:
    # https://github.com/dribnet/lapjv1
    using_lapjv1 = True
    import lapjv1

def real_glob(rglob):
    glob_list = braceexpand(rglob)
    files = []
    for g in glob_list:
        files = files + glob.glob(g)
    return sorted(files)

def get_image(path, input_shape):
    img = image.load_img(path, target_size=input_shape)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_average_color(path):
    c = scipy.misc.imread(path, mode='RGB').mean(axis=(0,1))
    # WTF indeed (this happens for black)
    if isinstance(c, numbers.Number):
        c = [c, c, c]
    return c

def get_image_list(input_glob, width, height):
    images = real_glob(input_glob)
    print("Found {} images".format(len(images)))
    if width is None:
        biggest_square = int(math.sqrt(len(images)))
        width, height = biggest_square, biggest_square
    max_images = width * height
    images = images[:max_images]
    num_images = len(images)
    if num_images == 0:
        print("Error: no images in {}".format(input_glob))
        sys.exit(0)
    print("Using {} images to build {}x{} montage".format(num_images, width, height))
    return images, num_images, width, height

def analyze_images_colors(images):
    # analyze images and grab activations
    colors = []
    for image_path in images:
        try:
            c = get_average_color(image_path)
        except Exception as e:
            print("Problem reading {}: {}".format(image_path, e))
            c = [0, 0, 0]
        # print(image_path, c)
        colors.append(c)
    return np.asarray(colors) / 255.0

def analyze_images(images):
    num_images = len(images)
    # make feature_extractor
    model = keras.applications.VGG16(weights='imagenet', include_top=True)
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
    input_shape = model.input_shape[1:3]
    # analyze images and grab activations
    activations = []
    for idx,image_path in enumerate(images):
        file_path = image_path
        img = get_image(file_path, input_shape);
        if img is not None:
            print("getting activations for %s %d/%d" % (image_path,idx,num_images))
            acts = feat_extractor.predict(img)[0]
            activations.append(acts)
    # run PCA firt
    print("Running PCA on %d images..." % len(activations))
    features = np.array(activations)
    pca = PCA(n_components=300)
    pca.fit(features)
    pca_features = pca.transform(features)
    return np.asarray(pca_features)

def run_tsne(input_glob, output_path, tsne_dimensions, tsne_perplexity,
        tsne_learning_rate, width, height, do_colors):
    images, num_images, width, height = get_image_list(input_glob, width, height)
    num_images = width * height
    avg_colors = analyze_images_colors(images)
    if do_colors:
        X = avg_colors
    else:
        X = analyze_images(images)
    print("Running t-SNE on {} images...".format(num_images))
    tsne = TSNE(n_components=tsne_dimensions, learning_rate=tsne_learning_rate, perplexity=tsne_perplexity, verbose=2).fit_transform(X)

    # make output directory if needed
    if output_path != '' and not os.path.exists(output_path):
        os.makedirs(output_path)

    # save data to json
    data = []
    for i,f in enumerate(images):
        point = [ (tsne[i,k] - np.min(tsne[:,k]))/(np.max(tsne[:,k]) - np.min(tsne[:,k])) for k in range(tsne_dimensions) ]
        data.append({"path":images[i], "point":point})
    with open(os.path.join(output_path, "points.json"), 'w') as outfile:
        json.dump(data, outfile)

    data2d = tsne
    data2d -= data2d.min(axis=0)
    data2d /= data2d.max(axis=0)
    plt.figure(figsize=(8, 8))
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.gca().invert_yaxis()
    # if debug colors...
    # colors = ['black'] * len(data2d)
    # colors[0] = 'red'
    # colors[1] = 'green'
    # colors[2] = 'blue'
    # colors[-1] = 'yellow'
    # if debug_colors:
    #     graph_colors = X / 255.0
    # else:
    #     graph_colors = colors
    grays = np.linspace(0, 0.8, len(data2d))
    plt.scatter(data2d[:,0], data2d[:,1], c=avg_colors, edgecolors='none', marker='o', s=24)  
    plt.savefig(os.path.join(output_path, "tsne.png"), bbox_inches='tight')

    xv, yv = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    grid = np.dstack((xv, yv)).reshape(-1, 2)
    # print(grid.shape)
    # print(data2d.shape)

    cost = distance.cdist(grid, data2d, 'sqeuclidean')
    cost = cost * (100000. / cost.max())

    if using_lapjv1:
        min_cost2, row_assigns2, col_assigns2 = lapjv1.lapjv1(cost)
    else:
        # note slightly different API
        row_assigns2, col_assigns2, min_cost2 = lapjv.lapjv(cost, verbose=True, force_doubles=False)
    grid_jv2 = grid[col_assigns2]
    # print(col_assigns2.shape)
    plt.figure(figsize=(8, 8))
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.gca().invert_yaxis()
    for start, end, c in zip(data2d, grid_jv2, avg_colors):
        plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                  color=c, head_length=0.01, head_width=0.01)
    plt.savefig(os.path.join(output_path, 'movement.png'), bbox_inches='tight')

    n_images = np.asarray(images)
    image_grid = n_images[row_assigns2]
    filelist = os.path.join(output_path, "filelist.txt")
    with open(filelist, "w") as text_file:
        for image in image_grid:
            text_file.write("\"{}\"\n".format(image))
        # for i in col_assigns2:
        #     text_file.write("{}\n".format(images[i]))

    command = "montage @{} -geometry +0+0 -tile {}x{} {}".format(filelist,
        width, height, os.path.join(output_path, "grid.jpg"))
    os.system(command)

def main():
    parser = argparse.ArgumentParser(description="Deep learning grid layout")
    parser.add_argument('--input-glob', default=None,
                        help="use file glob source of images")
    parser.add_argument('--output-path', 
                         help='path to where to put output files')
    parser.add_argument('--num-dimensions', default=2, type=int,
                        help='dimensionality of t-SNE points')
    parser.add_argument('--perplexity', default=30, type=int,
                        help='perplexity of t-SNE')
    parser.add_argument('--learning-rate', default=150, type=int,
                        help='learning rate of t-SNE')
    parser.add_argument('--do-colors', default=False, action='store_true',
                        help="Use average color as feature")
    parser.add_argument('--tile', default=None,
                        help="Grid size WxH (eg: 12x12)")
    args = parser.parse_args()
    width, height = None, None
    if args.tile is not None:
        width, height = map(int, args.tile.split("x"))
    run_tsne(args.input_glob, args.output_path, args.num_dimensions, 
             args.perplexity, args.learning_rate, width, height, args.do_colors)

if __name__ == '__main__':
    main()
