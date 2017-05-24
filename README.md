# Intelligent grid layout from a set of images.

# basics

You can get started with the [someflags dataset](https://github.com/vusd/smartgrid/releases/download/someflags/someflags.zip) - unzip in the datasets subdirectory. Then:

```bash
python smartgrid.py \
  --input-glob 'datasets/someflags/*.png' \
  --output-path outputs/flag_grid
```
Output is `outputs/flag_grid/grid.jpg`:
![default flag grid](https://cloud.githubusercontent.com/assets/945979/26386053/08cb8076-4098-11e7-8caa-9449cd241e85.jpg)

Output is a set of files in the provided directory. These files inclde images that show the original TSNE layout and grid assignments.

# options

## different models
The arrangement is based on a trained neural network, and many model options are available via keras. They are:
 * vgg16
 * vgg19
 * resnet50
 * inceptionv3
 * xception

In addition, the specific layer of the network to be used for feature extraction can be provided as well. And for inceptionv3, a pooling option is available. By default vgg16/fc2 is used.

For example:
```bash
python smartgrid.py \
  --model inceptionv3 \
  --pooling avg \
  --input-glob 'datasets/someflags/*.png' \
  --output-path outputs/flags_inception_avgpool
```

![inception avg pooling](https://cloud.githubusercontent.com/assets/945979/26386117/9afb5c3c-4098-11e7-96dc-2bc444a2c982.jpg)

Grouping by color is also possible by using the `color` or `color_lab` models:

```bash
python smartgrid.py \
  --model color \
  --input-glob 'datasets/someflags/*.png' \
  --output-path outputs/flag_grid_colors
```
Output is `outputs/flag_grid_colors/grid.jpg`:
![flag color grid](https://cloud.githubusercontent.com/assets/945979/26386050/08b8c170-4098-11e7-885f-787fd17e31b3.jpg)

## tile, aspect-ratio, left-right anchors

Optional command line arguements are also available that change the aspect ratio and influence the layout. For example:

```bash
python smartgrid.py \
  --tile 24x12 \
  --input-glob 'datasets/someflags/*.png' \
  --left-image 'datasets/someflags/FR.png' \
  --right-image 'datasets/someflags/NL.png' \
  --output-path outputs/someflags_horizvert_s10
```
![flag color grid](https://cloud.githubusercontent.com/assets/945979/26386049/08b85140-4098-11e7-8f80-d0158fd22b11.jpg)

This set of arguments creates a non-square grid and also suggests that the `FR.png` image (🇫🇷) should be laid out to the left of the `NL.png` image (🇳🇱). The left/right image flags also try to influence groupings by exaggerating the differences between these anchors (this stretching can be disabled by setting `--left-right-scale 0.0`).

The tile argument specifies the number of rows and columns. You can also specify `--aspect-ratio` to have the grid image fit a specific format.

## filtering, output format and filename

An experimental argument `--min-distance` has been added that will remove duplicates based on the distance apart in feature space. Additionally, the output file name can be overridden, and the file format will
be inferred from the name. So to output the flag grid in png format without duplicates:

```bash
python smartgrid.py \
  --aspect-ratio 1.778 \
  --input-glob 'datasets/someflags/*.png' \
  --min-distance 5 \
  --grid-file grid_min_dist_5.png \
  --output-path outputs/someflags_nodupes
```
Output this time is `outputs/someflags_nodupes/grid_min_dist_5.png`:
![filtered png file](https://cloud.githubusercontent.com/assets/945979/26386051/08cae440-4098-11e7-9fde-3ac0ad8ccbde.png)

Note the duplicates were removed (eg: there were 2 US flags before). The argument has to be fiddled with, but there is an output folder `rejects` which shows the duplicates that were found.

## rerunning, grid spacing, and visualizing link strength

Rerunning can be done more quickly by keeping the same output-path and adding the `--do-reload` flag. You probably also want to remove the `model`, `layer` arguments and perhaps change the `grid-file` output.

The `--grid-spacing` option puts space between elements on the grid. For example, `--grid-spacing 1` will add a one pixel border to the grid elements.

Additionally, there is an experimental option to use the grid spacing to visualize the strength between adjacent entries. For this you add `--show-links`.

Putting that together, we can reuse the result from the section above and output to a different filename showing link strength.

```bash
python smartgrid.py \
  --do-reload \
  --aspect-ratio 1.778 \
  --input-glob 'datasets/someflags/*.png' \
  --min-distance 5 \
  --show-links \
  --grid-spacing 24 \
  --grid-file grid_with_links.jpg \
  --output-path outputs/someflags_nodupes
```
![reload with spacing and links](https://cloud.githubusercontent.com/assets/945979/26386054/08cb8e54-4098-11e7-9fec-5fb6a553ac79.jpg)

In the current visualization, the closer the entries are in feature space the *thinner* the line between them (think of the line as a wall that wants to separate them). Also this run much faster because the neural net is no longer needed when `--do-reload` is used.

## imagemagick (for the big grids)

Extrememly large grids blow up because of a PIL memory limitation. In this case you can fallback
to using imagemagick (montage) to construct the grid. So if you have 4700 images to group perceptually
by color:

```bash
python smartgrid.py \
  --model color_lab \
  --use-imagemagick \
  --aspect-ratio 1.778 \
  --input-glob 'datasets/new_yorker/*.jpg' \
  --output-path outputs/ny_color_lab

# resize output with imagemagick
convert outputs/ny_color_lab/grid.jpg \
  -resize 5%  \
  outputs/ny_color_lab/grid_scaled.jpg
```
Go get a coffee. Then come back to find `outputs/ny_color_lab/grid_scaled.jpg`:
![huge grid shrunken](https://cloud.githubusercontent.com/assets/945979/26386052/08cb66e0-4098-11e7-8222-7ec9afaf50ed.jpg)

# dependencies

Currently requires keras 2.x, scipy, sklearn, matplotlib,
braceexpand, tqdm, and either [lapjv1](https://github.com/dribnet/lapjv1) (seems to work everywhere but sometimes hangs) or [lapjv](https://github.com/src-d/lapjv) (runs much faster for me and provides verbose output). Also requires imagemagick (montage) when using `--use-imagemagick` option.

# credits

Code originally adapted from genekogan's [tSNE-images.py](https://github.com/ml4a/ml4a-ofx/blob/master/scripts/tSNE-images.py) and kylemcdonald's [CloudToGrid](https://github.com/kylemcdonald/CloudToGrid)

# license

WTFPL