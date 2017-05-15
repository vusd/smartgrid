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
![flag color grid](https://github.com/vusd/smartgrid/releases/download/0.1.0/left_right_layout.jpg)

This set of arguments creates a non-square grid and also suggests that the `FR.png` image (🇫🇷) should be laid out to the left of the `NL.png` image (🇳🇱). These layout flags will also influence the groupings by exaggerating the differences between these anchors - in this case emphasizing shapes over colors (this stretching can be disabled by setting `--left-right-scale 0.0`).

Currently requires imagemagick (montage), keras 2.x, scipy, sklearn, matplotlib,
braceexpand, and either [lapjv](https://github.com/src-d/lapjv) or [lapjv1](https://github.com/dribnet/lapjv1).

Code adapted from @genekogan's [tSNE-images.py](https://github.com/ml4a/ml4a-ofx/blob/master/scripts/tSNE-images.py) and @kylemcdonald's [CloudToGrid](https://github.com/kylemcdonald/CloudToGrid)
