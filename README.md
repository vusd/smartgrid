Intelligent grid layout from a set of images.

You can get started with the [someflags dataset](https://github.com/vusd/smartgrid/releases/download/someflags/someflags.zip) - unzip in the datasets subdirectory. Then:

```bash
python smartgrid.py \
  --input-glob 'datasets/someflags/*.png' \
  --output-path outputs/flag_grid
```
Output is `outputs/flag_grid/all.jpg`:
![flag grid](https://github.com/vusd/smartgrid/releases/download/someflags/grid_someflags.jpg)

Grouping by color is also possible:
```bash
python smartgrid.py \
  --do-colors \
  --input-glob 'datasets/someflags/*.png' \
  --output-path outputs/flag_grid_colors
```
Output is `outputs/flag_grid_colors/all.jpg`:
![flag color grid](https://github.com/vusd/smartgrid/releases/download/someflags/grid_someflags_color.jpg)

Output is a set of files in the provided directory. These files inclde images that show the original TSNE layout and grid assignments. Optional command line arguements are also available that change the aspect ratio and influence the layout. For example:

```bash
python smartgrid.py \
  --tile 24x12 \
  --input-glob 'datasets/someflags/*.png' \
  --left-image 'datasets/someflags/FR.png' \
  --right-image 'datasets/someflags/NL.png' \
  --output-path outputs/someflags_horizvert_s10
```
![flag color grid](https://github.com/vusd/smartgrid/releases/download/extras/left_right_layout.jpg)

This set of arguments creates a non-square grid and also suggests that the `FR.png` image (🇫🇷) should be laid out to the left of the `NL.png` image (🇳🇱). The left/right image flags also try to influence groupings by exaggerating the differences between these anchors (this stretching can be disabled by setting `--left-right-scale 0.0`).

The tile argument specifies the number of rows and columns. You can also specify `--aspect-ratio` to have the grid image fit a specific format.

The arrangement is based on a trained neural network, and many model options are available via keras. They are:
 * vgg16
 * vgg19
 * resnet50
 * inceptionv3
 * xception

In addition, the specific layer of the network to be used for feature extraction can be provided as well. For example:
```bash
python smartgrid.py \
  --model vgg19 \
  --layer fc1 \
  --input-glob 'datasets/someflags/*.png' \
  --output-path outputs/flag_grid_colors_vgg19_fc1 \
  --aspect-ratio 1 \
  --output-path outputs/vgg_fc1_full_01_square
```

![flag square vgg19_fc1 grid](https://github.com/vusd/smartgrid/releases/download/extras/square_layout.jpg)

Currently requires imagemagick (montage), keras 2.x, scipy, sklearn, matplotlib,
braceexpand, tqdm, and either [lapjv](https://github.com/src-d/lapjv) or [lapjv1](https://github.com/dribnet/lapjv1).

Code adapted from @genekogan's [tSNE-images.py](https://github.com/ml4a/ml4a-ofx/blob/master/scripts/tSNE-images.py) and @kylemcdonald's [CloudToGrid](https://github.com/kylemcdonald/CloudToGrid)
