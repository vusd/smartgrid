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
using_python_lap = False
try:
    # https://github.com/src-d/lapjv
    import lapjv
except ImportError:
    try:
        # https://github.com/dribnet/python-lap/tree/rename_lap
        using_python_lap = True
        import lap
    except ImportError:
        print("Error: could not find lapjv or python-lap, cannot continue")
        sys.exit(1)

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

def get_average_color_classic(path, colorspace='rgb'):
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

def get_average_color(path, colorspace='rgb', subsampling=None):
    im = scipy.misc.imread(path, mode='RGB')
    w, h, c = im.shape
    colors = []
    if subsampling is None:
        subsampling = "1";
    if subsampling.endswith("+"):
        sample_from = int(subsampling[:-1])
        sample_downto = 0
    else:
        sample_from = int(subsampling)
        sample_downto = sample_from-1
    for gridsize in range(sample_from, sample_downto, -1):
        for y in range(gridsize):
            h1 = int(y*h/gridsize)
            h2 = int((y+1)*h/gridsize)
            for x in range(gridsize):
                w1 = int(x*w/gridsize)
                w2 = int((x+1)*w/gridsize)
                quadrant = im[w1:w2, h1:h2, :]

                if colorspace == 'lab':
                    c = color.rgb2lab(quadrant)
                    c = c.mean(axis=(0,1))
                else:
                    c = quadrant.mean(axis=(0,1))
                    c = c / 255.0

                # WTF indeed (this happens for black (rgb))
                if isinstance(c, numbers.Number):
                    c = [c, c, c]

                colors.append(c)

    return np.array(colors).flatten()

def read_file_list(filelist):
    lines = []
    with open(filelist) as file:
        for line in file:
            line = line.strip() #or someother preprocessing
            line = line.strip( '"' ) # remove quotes
            lines.append(line)
    return lines

def read_json_vectors(filename):
    """Return np array of vectors from json sources"""
    vectors = []
    with open(filename) as json_file:
        json_data = json.load(json_file)
    for v in json_data:
        vectors.append(v)
    print("Read {} vectors from {}".format(len(vectors), filename))
    np_array = np.array(vectors)
    return np_array

def get_image_list(input_glob):
    if input_glob.startswith('@'):
        images = read_file_list(input_glob[1:])
    else:
        images = real_glob(input_glob)
    num_images = len(images)
    print("Found {} images".format(num_images))
    return images

def set_grid_size(images, width, height, aspect_ratio, drop_to_fit):
    num_images = len(images)
    if width is None and aspect_ratio is None:
        # just have width == height
        max_side_extent = math.sqrt(num_images)
        if max_side_extent.is_integer() or drop_to_fit:
            width = int(max_side_extent)
            height = width
        else:
            width = int(max_side_extent) + 1
            print("Checking: ", width*(width-1), num_images)
            if width*(width-1) >= num_images:
                height = width-1
            else:
                height = width
    elif width is None:
        # sniff the aspect ratio of the first file
        with Image.open(images[0]) as img:
            im_width = img.size[0]
            im_height = img.size[1]
            tile_aspect_ratio =  im_width / im_height
        raw_height = math.sqrt((num_images * tile_aspect_ratio) / aspect_ratio)
        raw_width = num_images / raw_height
        int_height = int(raw_height)
        int_width = int(raw_width)
        if (raw_height.is_integer() and raw_width.is_integer()) or drop_to_fit:
            height = int_height
            width = int_width
            if not drop_to_fit:
                print("--> {} images fits exactly as {}x{}".format(num_images, width, height))
        else:
            if not raw_height.is_integer():
                int_height = int_height + 1
            if not raw_width.is_integer():
                int_width = int_width + 1
            if int_width*(int_height-1) >= num_images:
                width = int_width
                height = int_height-1
            else:
                width = int_width
                height = int_height
            print("--> {} images best fits as {}x{}".format(num_images, width, height))
        print("tile size is {}x{} so aspect of {:.3f} is {}x{} (final: {}x{})".format(
            im_width, im_height, aspect_ratio, width, height, width*im_width, height*im_height))

    num_grid_spaces = width * height
    if drop_to_fit:
        grid_images = images[:num_grid_spaces]
        num_images = len(grid_images)
    else:
        grid_images = images

    if num_grid_spaces < num_images:
        print("Error: {} images is too many for {}x{} grid.".format(num_images, width, height))
        sys.exit(0)
    elif num_images == 0:
        print("Error: no images in {}".format(input_glob))
        sys.exit(0)
    elif num_grid_spaces == 0:
        print("Error: no spaces for images")
        sys.exit(0)

    print("Using {} images to build {}x{} montage".format(num_images, width, height))
    return grid_images, width, height

def normalize_columns(rawpoints, low=0, high=1):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    scaled_points = high - (((high - low) * (maxs - rawpoints)) / rng)
    return scaled_points

def analyze_images_colors(images, colorspace='rgb', subsampling=None):
    # analyze images and grab activations
    colors = []
    for image_path in images:
        try:
            if subsampling is None:
                c = get_average_color_classic(image_path, colorspace)
            else:
                c = get_average_color(image_path, colorspace, subsampling)
        except Exception as e:
            print("Problem reading {}: {}".format(image_path, e))
            c = [0, 0, 0]
        # print(image_path, c)
        colors.append(c)
    # colors = normalize_columns(colors)
    return colors

def analyze_images(images, model_name, layer_name=None, pooling=None, do_crop=False, subsampling=None):
    if model_name == 'color_lab':
        return analyze_images_colors(images, colorspace='lab', subsampling=subsampling)
    elif model_name == 'color' or model_name == 'color_rgb':
        return analyze_images_colors(images, colorspace='rgb', subsampling=subsampling)

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
        model = keras.applications.InceptionV3(weights='imagenet', include_top=include_top, pooling=pooling)
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
    elif layer_name == "show" or layer_name == "list":
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
            if len(activations) == 0:
                print("Collecting vectors of size {}".format(acts.flatten().shape))
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

def read_list(output_path, output_file, numeric=False):
    filelist = os.path.join(output_path, output_file)
    lines = []
    with open(filelist) as file:
        for line in file:
            line = line.strip() #or someother preprocessing
            if numeric:
                lines.append(list(map(float, line.split(","))))
            else:
                lines.append(line)
    if numeric:
        return np.array(lines)
    else:
        return lines

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
                    link_right_height = max_link_size * (1.0 - cell[0])
                    oy = int(link_right_height / 2)
                    ldw = int(link_right_height)
                    im_array[(cy-oy):(cy-oy+ldw), cx:(cx+width), :] = 0
                if cell[1] > 0:
                    link_down_width = max_link_size * (1.0 - cell[1])
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

def filter_distance_min(images, X, min_distance, reject_dir=None):
    num_images = len(images)
    keepers = [True] * num_images
    cur_pos = 0
    assignments = []
    min_distance2 = min_distance * min_distance
    for i in range(num_images):
        if not keepers[i]:
            continue
        rejects = []
        assignments.append(i)
        cur_v = X[i]
        for j in range(i+1, num_images):
            if keepers[j]:
                # if np.linalg.norm(cur_v - X[j]) < min_distance:
                diff = cur_v - X[j]
                if np.dot(diff, diff) < min_distance2:
                    rejects.append(j)
                    keepers[j] = False
        if len(rejects) > 0:
            print("rejecting {} images similar to entry {}".format(len(rejects), i))
            if reject_dir:
                reject_grid = [images[i]]
                for ix in rejects:
                    reject_grid.append(images[ix])
                img = make_grid_image(reject_grid)
                reject_file_path = os.path.join(reject_dir,
                    "dist_{:04f}_{:03d}.jpg".format(min_distance, i))
                img.save(reject_file_path)


    print("Keeping {} of {} images".format(len(assignments), num_images))
    im_array = np.array(images)
    X_array = np.array(X)
    return im_array[assignments].tolist(), X_array[assignments]

def filter_distance_max(images, X, max_distance, reject_dir=None, max_group_size=1):
    num_images = len(images)
    keepers = [False] * num_images
    cur_pos = 0
    assignments = []
    max_distance2 = max_distance * max_distance
    for i in range(num_images):
        if keepers[i]:
            assignments.append(i)
            continue
        accepts = []
        cur_v = X[i]
        for j in range(i+1, num_images):
            if not keepers[j]:
                # if np.linalg.norm(cur_v - X[j]) < max_distance:
                diff = cur_v - X[j]
                if np.dot(diff, diff) < max_distance2:
                    keepers[i] = True
                    keepers[j] = True
                    accepts.append(j)

        if len(accepts) >= max_group_size:
            print("accepting {} images similar to entry {}".format(len(accepts), i))
            assignments.append(i)
            if reject_dir:
                reject_grid = [images[i]]
                for ix in accepts:
                    reject_grid.append(images[ix])
                img = make_grid_image(reject_grid)
                reject_file_path = os.path.join(reject_dir, 
                    "dist_{:04f}_{:03d}.jpg".format(max_distance, i))
                img.save(reject_file_path)


    print("Keeping {} of {} images".format(len(assignments), num_images))
    im_array = np.array(images)
    X_array = np.array(X)
    return im_array[assignments].tolist(), X_array[assignments]

def reduce_grid_targets(grid, num_grid_images):
    print("reducing grid from {} to {}".format(len(grid), num_grid_images))
    mean_point = np.mean(grid, axis=0)
    newList = grid - mean_point
    sort = np.sum(np.power(newList, 2), axis=1)
    indexed_order = sort.argsort()
    sorted_list = grid[indexed_order]
    return sorted_list[:num_grid_images], indexed_order

def run_prune(filelist, vectorlist):
    new_filelist = []
    new_vectorlist = []
    for i in range(len(vectorlist)):
        if vectorlist[i] is not None and os.path.exists(filelist[i]):
            new_filelist.append(filelist[i])
            new_vectorlist.append(vectorlist[i])
    print("Pruned filelist from {} to {} entries".format(len(filelist), len(new_filelist)))
    return new_filelist, np.array(new_vectorlist)

# in the future the clip_range could be smarter,
# like 1-4,100-200 etc.
# for now, just doing head
def run_clip(filelist, vectorlist, clip_range):
    clip_number = int(clip_range)
    new_filelist = filelist[:clip_number]
    new_vectorlist = vectorlist[:clip_number]
    return new_filelist, np.array(new_vectorlist)

def run_grid(input_glob, left_image, right_image, left_right_scale,
        output_path, tsne_dimensions, tsne_perplexity,
        tsne_learning_rate, width, height, aspect_ratio, drop_to_fit, fill_shade,
        vectors_file, do_prune, clip_range, subsampling,
        model, layer, pooling, do_crop, grid_file, use_imagemagick,
        grid_spacing, show_links, links_max_threshold,
        min_distance, max_distance, max_group_size, do_reload=False):

    # make output directory if needed
    if output_path != '' and not os.path.exists(output_path):
        os.makedirs(output_path)

    if do_reload:
        images = read_list(output_path, "image_files.txt", numeric=False)
        X = read_list(output_path, "image_vectors.txt", numeric=True)
        print("Reloaded {} images and {} vectors".format(len(images), X.shape))
        num_images = len(images)
        avg_colors = analyze_images_colors(images, 'rgb')
    else:
        ## compute width,weight from image list and provided defaults
        if input_glob is not None:
            images = get_image_list(input_glob)
            num_images = len(images)

        if vectors_file is not None:
            X = read_json_vectors(vectors_file)
        else:
            X = analyze_images(images, model, layer, pooling, do_crop, subsampling)

        if do_prune:
            images, X = run_prune(images, X)

        if clip_range:
            images, X = run_clip(images, X, clip_range)

        # save data
        write_list(images, output_path, "image_files.txt")
        write_list(X, output_path, "image_vectors.txt")

    avg_colors = analyze_images_colors(images, 'rgb')

    ## Lookup left/right images
    left_image_index = None
    right_image_index = None
    # scale X by left/right axis
    if left_image is not None and right_image is not None:
        left_image_index = index_from_substring(images, left_image)
        right_image_index = index_from_substring(images, right_image)

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
        reject_dir = os.path.join(output_path, "rejects_min")
        if reject_dir != '' and not os.path.exists(reject_dir):
            os.makedirs(reject_dir)
        images, X = filter_distance_min(images, X, min_distance, reject_dir)

    if max_distance is not None:
        reject_dir = os.path.join(output_path, "rejects_max")
        if reject_dir != '' and not os.path.exists(reject_dir):
            os.makedirs(reject_dir)
        images, X = filter_distance_max(images, X, max_distance, reject_dir, max_group_size)

    grid_images, width, height = set_grid_size(images, width, height, aspect_ratio, drop_to_fit)
    num_grid_images = len(grid_images)
    print("Compare: {} and {}".format(num_grid_images, width*height))

    # this line is a hack for now
    X = np.asarray(X[:num_grid_images])

    print("SO X {}".format(X.shape))
    print("Running t-SNE on {} images...".format(num_grid_images))
    tsne = TSNE(n_components=tsne_dimensions, learning_rate=tsne_learning_rate, perplexity=tsne_perplexity, verbose=2).fit_transform(X)

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

    # this is an experimental section where left/right image can be given
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

    # TSNE is done, setup layout for grid assignment
    max_width, max_height = 1, 1
    if (width > height):
        max_height = height / width
    elif(width < height):
        max_width = width / height
    xv, yv = np.meshgrid(np.linspace(0, max_width, width), np.linspace(0, max_height, height))
    grid = np.dstack((xv, yv)).reshape(-1, 2)
    # this strange step removes corners
    grid, indexed_lookup = reduce_grid_targets(grid, num_grid_images)
    # print("G", grid.shape)
    # print("D2D", data2d.shape)

    cost = distance.cdist(grid, data2d, 'euclidean')
    # cost = distance.cdist(grid, data2d, 'sqeuclidean')
    cost = cost * (100000. / cost.max())
    # print("C", cost.shape, cost[0][0])

    if using_python_lap:
        print("Starting assignment (this can take a few minutes)")
        min_cost2, row_assigns2, col_assigns2 = lap.lapjv(cost)
        print("Assignment complete")
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


    num_grid_spaces = len(indexed_lookup)
    num_actual_images = len(row_assigns2)
    num_missing = num_grid_spaces - num_actual_images
    using_placeholder = False

    if num_missing > 0:
        # makde a note that placeholder is in use
        using_placeholder = True

        # add a blank entry to the vectors
        _, v_len = X.shape
        X2 = np.append(X, [np.zeros(v_len)], axis=0)
        print("Updating vectors from {} to {}".format(X.shape, X2.shape))
        X = X2

        # add blank entry to images
        # sniff the aspect ratio of the first file
        with Image.open(grid_images[0]) as img:
            im_width = img.size[0]
            im_height = img.size[1]

        im_array = np.full([im_height, im_width, 3], [fill_shade, fill_shade, fill_shade]).astype(np.uint8)
        # im_array = np.zeros([im_width, im_height, 3]).astype(np.uint8)
        blank_img = Image.fromarray(im_array)
        blank_image_path = os.path.join(output_path, "blank.png")
        blank_img.save(blank_image_path)
        blank_index = len(grid_images)
        grid_images.append(blank_image_path)

        # now grow row assignments, giving all remaining to new blanks
        residuals = np.full([num_missing], blank_index)
        row_assigns2 = np.append(row_assigns2, residuals)

    reverse_lookup = np.zeros(num_grid_spaces, dtype=int)
    reverse_lookup[indexed_lookup] = np.arange(num_grid_spaces)

    image_indexes = row_assigns2[reverse_lookup]

    n_images = np.asarray(grid_images)
    image_grid = n_images[image_indexes]
    montage_filelist = write_list(image_grid, output_path, 
        "montage_{}x{}.txt".format(width, height), quote=True)
    grid_file_path = os.path.join(output_path, grid_file)
    grid_im_file_path = os.path.join(output_path, "{}".format(grid_file))
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
        links = None
        if show_links:
            links = []
            img_grid_vectors = X[image_indexes]
            for r in range(height):
                row = []
                links.append(row)
                for c in range(width):
                    idx = r * width + c
                    cur_v = img_grid_vectors[idx]
                    if c < width - 1:
                        left_v = img_grid_vectors[idx+1]
                        if using_placeholder and (not cur_v.any() or not left_v.any()):
                            dist_left = -1
                        else:
                            dist_left = np.linalg.norm(cur_v - left_v)
                    else:
                        dist_left = -1
                    if r < height - 1:
                        down_v = img_grid_vectors[idx+width]
                        if using_placeholder and (not cur_v.any() or not down_v.any()):
                            dist_down = -1
                        else:
                            dist_down = np.linalg.norm(cur_v - down_v)
                    else:
                        dist_down = -1
                    cell = [dist_left, dist_down]
                    row.append(cell)
            links = np.array(links)
            # normalize to 0-1
            if links_max_threshold is not None:
                num_removed = (links > links_max_threshold).sum()
                links[links > links_max_threshold] = -1
                num_left = (links > 0).sum()
                print("removed {} links, {} left".format(num_removed, num_left))
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
    parser.add_argument('--vectors', default=None,
                        help="read vectors directly instead of running model")
    parser.add_argument('--do-prune', default=False, action='store_true',
                        help="Prune filelist filtering if vectors missing")
    parser.add_argument('--clip-range', default=None,
                        help="only show range of images given (eg: 100)")
    parser.add_argument('--model', default=None,
                        help="model to use, one of: vgg16 vgg19 resnet50 inceptionv3 xception")
    parser.add_argument('--layer', default=None,
                        help="optional override to set custom model layer")
    parser.add_argument('--pooling', default=None,
                        help="optional override to control inceptionv3 pooling (avg or max)")
    parser.add_argument('--subsampling', default=None,
                        help="subsampling specifier for tiles (for some models). eg: 2+")
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
    parser.add_argument('--drop-to-fit', default=False, action='store_true',
                        help="Drop extra images to fit to aspect ratio")
    parser.add_argument('--fill-shade', default=0, type=int,
                        help='shade of gray for filling in blanks')
    parser.add_argument('--use-imagemagick', default=False, action='store_true',
                        help="generate grid using imagemagick (montage)")
    parser.add_argument('--tile', default=None,
                        help="Grid size WxH (eg: 12x12)")
    parser.add_argument('--grid-spacing', default=0, type=int,
                        help='whitespace between images in grid')
    parser.add_argument('--show-links', default=False, action='store_true',
                        help="visualize link strength in whitespace")
    parser.add_argument('--links-max-threshold', default=None, type=float,
                        help="drop links past this threshold")
    parser.add_argument('--aspect-ratio', default=None, type=float,
                        help="Instead of square, fit image to given aspect ratio")
    parser.add_argument('--min-distance', default=None, type=float,
                        help="Removed duplicates based on distance")
    parser.add_argument('--max-distance', default=None, type=float,
                        help="Removes items if they are beyond max from all others")
    parser.add_argument('--max-group-size', default=1, type=int,
                        help='when max-distance, minimum number of additional members')
    parser.add_argument('--do-reload', default=False, action='store_true',
                        help="Reload file list and vectors from saved state")
    args = parser.parse_args()
    width, height = None, None
    if args.tile is not None:
        width, height = map(int, args.tile.split("x"))
    if args.model is None and args.layer is None:
        model = "vgg16"
        layer = "fc2"
    elif args.model is None:
        model = "vgg16"
        layer = args.layer
    else:
        model = args.model
        layer = args.layer
    # this obviously needs refactoring
    run_grid(args.input_glob, args.left_image, args.right_image, args.left_right_scale,
             args.output_path, args.num_dimensions, 
             args.perplexity, args.learning_rate, width, height, args.aspect_ratio,
             args.drop_to_fit, args.fill_shade, args.vectors, args.do_prune, args.clip_range,
             args.subsampling,
             model, layer, args.pooling, args.do_crop, args.grid_file, args.use_imagemagick,
             args.grid_spacing, args.show_links, args.links_max_threshold,
             args.min_distance, args.max_distance,
             args.max_group_size, args.do_reload)

if __name__ == '__main__':
    main()
