Intelligent grid layout from a set of images.

Example with flags:

```bash
python smartgrid.py \
  --input-glob '/path/to/images/*.png' \
  --output-path outputs/flag_grid
```

Grouping by color is also possible:
```bash
python smartgrid.py \
  --do-colors \
  --input-glob '/path/to/images/*.png' \
  --output-path outputs/flag_grid_colors
```

Output is a set of files in the provided directory.

Currently requires keras 2.x, scipy, sklearn, matplotlib,
braceexpand, and [lapjv](https://github.com/src-d/lapjv).

Code adapted from @genekogan's [tSNE-images.py](https://github.com/ml4a/ml4a-ofx/blob/master/scripts/tSNE-images.py) and @kylemcdonald's [CloudToGrid](https://github.com/kylemcdonald/CloudToGrid)
