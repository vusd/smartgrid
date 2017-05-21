import argparse
import sys
import numpy as np
import json
import os
from os.path import isfile, join
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
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

def center_crop(img, target_size):
     width, height = img.size
     smaller = width
     if height < width:
         smaller = height

     # TODO: this might be off by one
     left = np.ceil((width - smaller)/2.)
     top = np.ceil((height - smaller)/2.)
     right = np.floor((width + smaller)/2.)
     bottom = np.floor((height + smaller)/2.)
     img = img.crop((left, top, right, bottom))
     # print("resizing from {} to {}".format([width, height], target_size))
     img = img.resize(target_size)
     return img

def get_image(path, input_shape, do_crop=False):
    if do_crop:
        # cropping version
        img = image.load_img(path)
        # print(path)
        img = center_crop(img, target_size=input_shape)
    else:
        # scaling version
        img = image.load_img(path, target_size=input_shape)

    # img.save("sized.png")
    # print("DONE")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

def get_average_color(path, colorspace='rgb'):
    c = scipy.misc.imread(path, mode='RGB')
    if colorspace == 'lab':
        # print("CONVERTING TO LAB")
        # old_color = c
        c = color.rgb2lab(c)
        # print("Converted from {} to {}".format(old_color[0], c[0]))
        c = c.mean(axis=(0,1))
    else:
        c = c.mean(axis=(0,1))
        c = c / 255.0

    # WTF indeed (this happens for black (rgb))
    if isinstance(c, numbers.Number):
        c = [c, c, c]
    return c

def get_image_list(input_glob):
    images = real_glob(input_glob)
    num_images = len(images)
    print("Found {} images".format(num_images))
    return images

def set_grid_size(images, width, height, aspect_ratio):
    num_images = len(images)
    if width is None and aspect_ratio is None:
        # just have width == height
        biggest_square = int(math.sqrt(num_images))
        width, height = biggest_square, biggest_square
    elif width is None:
        # sniff the aspect ratio of the first file
        with Image.open(images[0]) as img:
            im_width = img.size[0]
            im_height = img.size[1]
            tile_aspect_ratio =  im_width / im_height
        height = int(math.sqrt((num_images * tile_aspect_ratio) / aspect_ratio))
        width = int(num_images / height)
        print("tile size is {}x{} so aspect of {:.3f} is {}x{} (final: {}x{})".format(
            im_width, im_height, aspect_ratio, width, height, width*im_width, height*im_height))

    num_grid_images = width * height
    if num_grid_images > num_images:
        print("Error: {} images is not enough for {}x{} grid.".format(num_images, width, height))
        sys.exit(0)
    elif num_grid_images == 0:
        print("Error: no images in {}".format(input_glob))
        sys.exit(0)

    grid_images = images[:num_grid_images]
    print("Using {} images to build {}x{} montage".format(num_images, width, height))
    return grid_images, width, height

def normalize_columns(rawpoints, low=0, high=1):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    scaled_points = high - (((high - low) * (maxs - rawpoints)) / rng)
    return scaled_points

def analyze_images_colors(images, colorspace='rgb'):
    # analyze images and grab activations
    colors = []
    for image_path in images:
        try:
            c = get_average_color(image_path, colorspace)
        except Exception as e:
            print("Problem reading {}: {}".format(image_path, e))
            c = [0, 0, 0]
        # print(image_path, c)
        colors.append(c)
    # colors = normalize_columns(colors)
    return colors

def analyze_images(images, model_name, layer_name=None, do_crop=False):
    if model_name == 'color_lab':
        return analyze_images_colors(images, colorspace='lab')
    elif model_name == 'color' or model_name == 'color_rgb':
        return analyze_images_colors(images, colorspace='rgb')

    num_images = len(images)

    preprocess_input = keras.applications.imagenet_utils.preprocess_input
    input_shape = (224, 224)
    include_top = (layer_name is not None)
    # make feature_extractor
    if model_name == 'vgg16':
        model = keras.applications.VGG16(weights='imagenet', include_top=include_top)
    elif model_name == 'vgg19':
        model = keras.applications.VGG19(weights='imagenet', include_top=include_top)
    elif model_name == 'resnet50':
        model = keras.applications.ResNet50(weights='imagenet', include_top=include_top)
    elif model_name == 'inceptionv3':
        # todo: add support for different "pooling" options
        model = keras.applications.InceptionV3(weights='imagenet', include_top=include_top)
    elif model_name == 'xception':
        model = keras.applications.Xception(weights='imagenet', include_top=include_top)
    else:
        print("Error: model {} not found".format(model_name))
        sys.exit(1)

    if model_name == 'inceptionv3' or model_name == 'xception':
        preprocess_input = keras.applications.inception_v3.preprocess_input
        input_shape = (299, 299)

    if layer_name is None:
        feat_extractor = model
    elif layer_name == "show":
        for i,layer in enumerate(model.layers):
            print("{} layer {:03d}: {}".format(model_name, i, layer.name))
        sys.exit(0)
    else:
        feat_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # analyze images and grab activations
    activations = []
    for idx in tqdm(range(len(images))):
        file_path = images[idx]
        img = get_image(file_path, input_shape, do_crop);
        if img is not None:
            # preprocess
            img = preprocess_input(img)
            # print("getting activations for %s %d/%d" % (file_path,idx,num_images))
            acts = feat_extractor.predict(img)[0]
            activations.append(acts.flatten())
    # run PCA firt
    features = np.array(activations)
    print("Running PCA on features: {}".format(features.shape))
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

def index_from_substring(images, substr):
    index = None
    for i in range(len(images)):
        # print("{} and {} and {}".format(images[i], substr, images[i].find(substr)))
        if images[i].find(substr) != -1:
            if index is None:
                index = i
            else:
                raise ValueError("The substring {} is ambiguious: {} and {}".format(
                    substr, images[index], images[i]))
    if index is None:
        raise ValueError("The substring {} was not found in {} images".format(substr, len(images)))
    else:
        print("Resolved {} to image {}".format(substr, images[index]))
    return index

def write_list(list, output_path, output_file, quote=False):
    filelist = os.path.join(output_path, output_file)
    with open(filelist, "w") as text_file:
        for item in list:
            if isinstance(item, np.ndarray):
                text_file.write("{}\n".format(",".join(map(str,item))))
            elif quote:
                text_file.write("\"{}\"\n".format(item))
            else:
                text_file.write("{}\n".format(item))
    return filelist

def make_grid_image(filelist, cols=None, rows=None, spacing=0, links=None):
    """Convert an image grid to a single image"""
    N = len(filelist)

    with Image.open(filelist[0]) as img:
        width = img.size[0]
        height = img.size[1]
        if width > height:
            max_link_size = int(1.0 * height)
        else:
            max_link_size = int(1.0 * width)

    if rows == None:
        sq_num = math.sqrt(N)
        sq_dim = int(sq_num)
        if sq_num != sq_dim:
            sq_dim = sq_dim + 1
        rows = sq_dim
        cols = sq_dim

    total_height = rows * height
    total_width  = cols * width

    total_height = total_height + spacing * (rows - 1)
    total_width  = total_width + spacing * (cols - 1)

    im_array = np.zeros([total_height, total_width, 3]).astype(np.uint8)
    im_array.fill(255)

    if links is not None:
        print("Rows: {}".format(len(links)))
        for r in range(len(links)):
            row = links[r]
            for c in range(len(row)):
                cell = row[c]
                offset_y, offset_x = r*height+spacing*r, c*width+spacing*c
                cy = int(offset_y + height / 2)
                cx = int(offset_x + width / 2)
                if cell[0] > 0:
                    link_right_height = max_link_size * cell[0]
                    oy = int(link_right_height / 2)
                    ldw = int(link_right_height)
                    im_array[(cy-oy):(cy-oy+ldw), cx:(cx+width), :] = 0
                if cell[1] > 0:
                    link_down_width = max_link_size * cell[1]
                    ox = int(link_down_width / 2)
                    lrw = int(link_down_width)
                    im_array[cy:(cy+height), (cx-ox):(cx-ox+lrw), :] = 0

    for i in range(rows*cols):
        if i < N:
            r = i // cols
            c = i % cols

            with Image.open(filelist[i]) as img:
                rgb_im = img.convert('RGB')
                offset_y, offset_x = r*height+spacing*r, c*width+spacing*c
                im_array[offset_y:(offset_y+height), offset_x:(offset_x+width), :] = rgb_im

    return Image.fromarray(im_array)

def filter_distance(images, X, min_distance, reject_dir=None):
    num_images = len(images)
    keepers = [True] * num_images
    cur_pos = 0
    assignments = []
    for i in range(num_images):
        if not keepers[i]:
            continue
        rejects = []
        assignments.append(i)
        cur_v = X[i]
        for j in range(i+1, num_images):
            if keepers[j]:
                if np.linalg.norm(cur_v - X[j]) < min_distance:
                    rejects.append(j)
                    keepers[j] = False
        if len(rejects) > 0:
            print("rejecting {} images similar to entry {}".format(len(rejects), i))
            if reject_dir:
                reject_grid = [images[i]]
                for ix in rejects:
                    reject_grid.append(images[ix])
                img = make_grid_image(reject_grid)
                reject_file_path = os.path.join(reject_dir, "reject_{:03d}.jpg".format(i))
                img.save(reject_file_path)


    print("Keeping {} of {} images".format(len(assignments), num_images))
    im_array = np.array(images)
    X_array = np.array(X)
    return im_array[assignments], X_array[assignments]

def run_grid(input_glob, left_image, right_image, left_right_scale,
        output_path, tsne_dimensions, tsne_perplexity,
        tsne_learning_rate, width, height, aspect_ratio,
        model, layer, do_crop, grid_file, use_imagemagick,
        grid_spacing, show_links, min_distance):

    ## compute width,weight from image list and provided defaults
    if input_glob is not None:
        images = get_image_list(input_glob)
        num_images = len(images)

    ## Lookup left/right images
    left_image_index = None
    right_image_index = None
    # scale X by left/right axis
    if left_image is not None and right_image is not None:
        left_image_index = index_from_substring(images, left_image)
        right_image_index = index_from_substring(images, right_image)

    avg_colors = analyze_images_colors(images, 'rgb')
    X = analyze_images(images, model, layer, do_crop)

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

    # TODO: filtering here
    if min_distance is not None:
        reject_dir = os.path.join(output_path, "rejects")
        if reject_dir != '' and not os.path.exists(reject_dir):
            os.makedirs(reject_dir)
        images, X = filter_distance(images, X, min_distance, reject_dir)

    grid_images, width, height = set_grid_size(images, width, height, aspect_ratio)
    num_grid_images = len(grid_images)

    # this line is a hack for now
    X = np.asarray(X[:num_grid_images])
    print("SO X {}".format(X.shape))
    print("Running t-SNE on {} images...".format(num_grid_images))
    tsne = TSNE(n_components=tsne_dimensions, learning_rate=tsne_learning_rate, perplexity=tsne_perplexity, verbose=2).fit_transform(X)

    # make output directory if needed
    if output_path != '' and not os.path.exists(output_path):
        os.makedirs(output_path)

    # save data
    write_list(images, output_path, "image_files.txt")
    write_list(X, output_path, "image_vectors.txt")
    data = []
    for i,f in enumerate(grid_images):
        point = [ (tsne[i,k] - np.min(tsne[:,k]))/(np.max(tsne[:,k]) - np.min(tsne[:,k])) for k in range(tsne_dimensions) ]
        data.append({"path":grid_images[i], "point":point})
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

    write_list(data2d, output_path, "tsne_coords.txt")

    max_width, max_height = 1, 1
    if (width > height):
        max_height = height / width
    elif(width < height):
        max_width = width / height
    xv, yv = np.meshgrid(np.linspace(0, max_width, width), np.linspace(0, max_height, height))
    grid = np.dstack((xv, yv)).reshape(-1, 2)
    # print(grid.shape)
    # print(data2d.shape)

    cost = distance.cdist(grid, data2d, 'euclidean')
    # cost = distance.cdist(grid, data2d, 'sqeuclidean')
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

    n_images = np.asarray(grid_images)
    image_grid = n_images[row_assigns2]
    montage_filelist = write_list(image_grid, output_path, 
        "montage_{}x{}.txt".format(width, height), quote=True)
    grid_file_path = os.path.join(output_path, grid_file)
    grid_im_file_path = os.path.join(output_path, "im_{}".format(grid_file))
    left_right_path = os.path.join(output_path, "left_right.jpg")
    if use_imagemagick:
        command = "montage @{} -geometry +0+0 -tile {}x{} {}".format(
            montage_filelist, width, height, grid_im_file_path)
        # print("running imagemagick montage: {}".format(command))
        os.system(command)

        # if left_image_index is not None:
        #     command = "montage '{}' '{}' -geometry +0+0 -tile 2x1 {}".format(
        #         images[left_image_index], images[right_image_index], left_right_path)
        #     os.system(command)

    else:
        # image vectors are in X
        img_grid_vectors = X[row_assigns2]
        links = None
        if show_links:
            links = []
            for r in range(height):
                row = []
                links.append(row)
                for c in range(width):
                    idx = r * width + c
                    cur_v = img_grid_vectors[idx]
                    if c < width - 1:
                        left_v = img_grid_vectors[idx+1]
                        dist_left = np.linalg.norm(cur_v - left_v)
                    else:
                        dist_left = -1
                    if r < height - 1:
                        down_v = img_grid_vectors[idx+width]
                        dist_down = np.linalg.norm(cur_v - down_v)
                    else:
                        dist_down = -1
                    cell = [dist_left, dist_down]
                    row.append(cell)
            links = np.array(links)
            # normalize to 0-1
            links_max = np.amax(links)
            valid_vals = np.where(links > 0)
            links_min = np.amin(links[valid_vals])
            print("Normalizing to {}/{}".format(links_min, links_max))
            links = ((links - links_min) / (links_max - links_min))
            print("Links is {}".format(links.shape))
        img = make_grid_image(image_grid, width, height, grid_spacing, links)
        img.save(grid_file_path)
        if left_image_index is not None:
            img = make_grid_image([grid_images[left_image_index], grid_images[right_image_index]], 2, 1, 1)
            img.save(left_right_path)


def main():
    parser = argparse.ArgumentParser(description="Deep learning grid layout")
    parser.add_argument('--input-glob', default=None,
                        help="use file glob source of images")
    parser.add_argument('--left-image', default=None,
                        help="use file as example of left")
    parser.add_argument('--right-image', default=None,
                        help="use file as example of right")
    parser.add_argument('--model', default='vgg16',
                        help="model to use, one of: vgg16 vgg19 resnet50 inceptionv3 xception")
    parser.add_argument('--layer', default=None,
                        help="optional override to set custom model layer")
    parser.add_argument('--left-right-scale', default=4.0, type=float,
                        help="scaling factor for left-right axis")
    parser.add_argument('--output-path', 
                         help='path to where to put output files')
    parser.add_argument('--grid-file', default="grid.jpg",
                         help='name (and format) of grid output file')
    parser.add_argument('--num-dimensions', default=2, type=int,
                        help='dimensionality of t-SNE points')
    parser.add_argument('--perplexity', default=30, type=int,
                        help='perplexity of t-SNE')
    parser.add_argument('--learning-rate', default=150, type=int,
                        help='learning rate of t-SNE')
    parser.add_argument('--do-crop', default=False, action='store_true',
                        help="Center crop instead of scale")
    parser.add_argument('--use-imagemagick', default=False, action='store_true',
                        help="generate grid using imagemagick (montage)")
    parser.add_argument('--tile', default=None,
                        help="Grid size WxH (eg: 12x12)")
    parser.add_argument('--grid-spacing', default=0, type=int,
                        help='whitespace between images in grid')
    parser.add_argument('--show-links', default=False, action='store_true',
                        help="visualize link strength in whitespace")
    parser.add_argument('--aspect-ratio', default=None, type=float,
                        help="Instead of square, fit image to given aspect ratio")
    parser.add_argument('--min-distance', default=None, type=float,
                        help="Removed duplicates based on distance")
    args = parser.parse_args()
    width, height = None, None
    if args.tile is not None:
        width, height = map(int, args.tile.split("x"))
    # this obviously needs refactoring
    run_grid(args.input_glob, args.left_image, args.right_image, args.left_right_scale,
             args.output_path, args.num_dimensions, 
             args.perplexity, args.learning_rate, width, height, args.aspect_ratio,
             args.model, args.layer, args.do_crop, args.grid_file, args.use_imagemagick,
             args.grid_spacing, args.show_links, args.min_distance)

if __name__ == '__main__':
    main()
