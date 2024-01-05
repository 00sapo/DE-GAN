# DE-GAN: A Conditional Generative Adversarial Network for Document deblurment

This is a fork of the [original repository](https://github.com/dali92002/DE-GAN) of the
DE-GAN model. It provides a simple API and CLI for using the model installable via
`pip`. Note that while neural models do most of the old-school hand-crafted image
processing they can still be improved by some custom modification. Pre- and
post-processing the images is recommended for optimizing the results to your use-case.

## Installation

`pip install git+https://github.com/00sapo/degan.git`

For CLI usage, I recommend using `pipx`: `pipx install git+https://githib.com/00sapo/degan.git`

For installing `pipx`, please refer to the [official documentation](https://pipx.pypa.io/stable/).

## Usage

### API

```python
from degan import DEGAN

# use the official weights distributed by the authors and provided within the pip package:
model = DEGAN()
model.binarize(image)
# input images should be grayscale float-32 numpy arrays with values in range [0, 1]
# You can use load_image and write_image for loading/writing to/from paths
from degan import load_image, write_image

# or use your own weights:
model = DEGAN(bin_weights='path/to/binary/weights.h5', deb_weights='path/to/deblur/weights.h5', wat_weights='path/to/unwatermark/weights.h5')

# the following loads the weights and run the inference:
binarized_image = model.binarize(image)
# and now it won't reload the weights but runs the inference on the same model:
binarized_image_2 = model.binarize(another_image)
# or you can force the loading of weights when you need it:
model.load_weights()
# you can also instantiate only certain models:
model = DEGAN(deblurred_image=None, deb_weights=None)
model.load_weights()

# similar for deblur and unwatermark
deblurred_image = model.deblur(image)
watermark_removed_image = model.unwatermark(image)

# you can also compute the PSNR metric:
from degan import psnr
psnr_value = psnr(image, binarized_image)
```

For training custom weights, please refer to the [original repository](https://github.com/dali92002/DE-GAN).

### CLI

**Note: for using as a CLI, consider installing via `pipx`**

- Default weights: `degan binarize image.png`
- Custom weights: `degan binarize image.png --out_dir ./out_dir/ --bin_weights path/to/binary/weights.h5`
- Other subcommands: `degan deblur`, `degan unwatermark`
- All options: `degan --help`, `degan - --help`

## Citation

- If this work was useful for you, please cite it as the original authors' publication:

```
@ARTICLE{Souibgui2020,
author={Mohamed Ali Souibgui and Yousri Kessentini},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
title={DE-GAN: A Conditional Generative Adversarial Network for Document deblurment},
year={2020},
doi={10.1109/TPAMI.2020.3022406}}
```
