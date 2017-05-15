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


def fit_to_unit_square(points, width, height):
    x_scale = 1.0
    y_scale = 1.0
    if (width > height):
        y_scale = height / width
    elif(width < height):
        x_scale = width / height
    points -= points.min(axis=0)
    points /= points.max(axis=0)
    points = points * [x_scale, y_scale]
    return points

def run_tsne(input_glob, left_image, right_image, left_right_scale,
        output_path, tsne_dimensions, tsne_perplexity,
        tsne_learning_rate, width, height, do_colors):
    images, num_images, width, height = get_image_list(input_glob, width, height)

    left_image_index = None
    right_image_index = None
    # scale X by left/right axis
    if left_image is not None and right_image is not None:
        left_image_index = images.index(left_image)
        right_image_index = images.index(right_image)

    num_images = width * height
    avg_colors = analyze_images_colors(images)
    if do_colors:
        X = avg_colors
    else:
        X = analyze_images(images)

    if left_image_index is not None:
        # todo: confirm this is how to stretch by a vector
        lr_vector = X[right_image_index] - X[left_image_index]
        lr_vector = lr_vector / np.linalg.norm(lr_vector)
        X_new = np.zeros_like(X)
        for i in range(len(X)):
            len_x = np.linalg.norm(X[i])
            norm_x = X[i] / len_x
            scale_factor = 1.0 + left_right_scale * (1.0 + np.dot(norm_x, lr_vector))
            new_length = len_x * scale_factor
            # print("Vector {}: length went from {} to {}".format(i, len_x, new_length))
            X_new[i] = new_length * norm_x
        X = X_new

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

    if left_image_index is not None:
        data2d = fit_to_unit_square(tsne, 1, 1)
    else:
        data2d = fit_to_unit_square(tsne, width, height)
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
    if left_image_index is not None:
        plt.scatter(data2d[left_image_index:left_image_index+1,0],
            data2d[left_image_index:left_image_index+1,1],
            facecolors='none', edgecolors='r', marker='o', s=24*3)
        plt.scatter(data2d[right_image_index:right_image_index+1,0],
            data2d[right_image_index:right_image_index+1,1],
            facecolors='none', edgecolors='g', marker='o', s=24*3)
    plt.savefig(os.path.join(output_path, "tsne.png"), bbox_inches='tight')

    if left_image_index is not None:
        origin = data2d[left_image_index]
        data2d = data2d - origin
        dest = data2d[right_image_index]
        x_axis = np.array([1, 0])
        theta = np.arctan2(dest[1],dest[0])
        print("Spin angle is {}".format(np.rad2deg(theta)))
        # theta = np.deg2rad(90)
        # print("Spin angle is {}".format(np.rad2deg(theta)))
        # # http://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
        a_c, a_s = np.cos(theta), np.sin(theta)
        R = np.matrix([[a_c, -a_s], [a_s, a_c]])
        data2d = np.array(data2d * R)
        # print("IS: ", data2d.shape)
        data2d = fit_to_unit_square(data2d, width, height)

        # TODO: this is a nasty cut-n-paste of above with different filename
        plt.figure(figsize=(8, 8))
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.gca().invert_yaxis()

        plt.scatter(data2d[:,0], data2d[:,1], c=avg_colors, edgecolors='none', marker='o', s=24)
        if left_image_index is not None:
            plt.scatter(data2d[left_image_index:left_image_index+1,0],
                data2d[left_image_index:left_image_index+1,1],
                facecolors='none', edgecolors='r', marker='o', s=48)
            plt.scatter(data2d[right_image_index:right_image_index+1,0],
                data2d[right_image_index:right_image_index+1,1],
                facecolors='none', edgecolors='g', marker='o', s=48)
        plt.savefig(os.path.join(output_path, "tsne_spun.png"), bbox_inches='tight')

    max_width, max_height = 1, 1
    if (width > height):
        max_height = height / width
    elif(width < height):
        max_width = width / height
    xv, yv = np.meshgrid(np.linspace(0, max_width, width), np.linspace(0, max_height, height))
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
        if left_image_index is not None:
            plt.scatter(data2d[left_image_index:left_image_index+1,0],
                data2d[left_image_index:left_image_index+1,1],
                facecolors='none', edgecolors='r', marker='o', s=48)
            plt.scatter(data2d[right_image_index:right_image_index+1,0],
                data2d[right_image_index:right_image_index+1,1],
                facecolors='none', edgecolors='g', marker='o', s=48)
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
    parser.add_argument('--left-image', default=None,
                        help="use file as example of left")
    parser.add_argument('--right-image', default=None,
                        help="use file as example of right")
    parser.add_argument('--left-right-scale', default=4.0, type=float,
                        help="scaling factor for left-right axis")
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
    run_tsne(args.input_glob, args.left_image, args.right_image, args.left_right_scale,
             args.output_path, args.num_dimensions, 
             args.perplexity, args.learning_rate, width, height, args.do_colors)

if __name__ == '__main__':
    main()
