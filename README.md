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

Output is a set of files in the provided directory.

Currently requires imagemagick (montage), keras 2.x, scipy, sklearn, matplotlib,
braceexpand, and [lapjv](https://github.com/src-d/lapjv).

Code adapted from @genekogan's [tSNE-images.py](https://github.com/ml4a/ml4a-ofx/blob/master/scripts/tSNE-images.py) and @kylemcdonald's [CloudToGrid](https://github.com/kylemcdonald/CloudToGrid)
