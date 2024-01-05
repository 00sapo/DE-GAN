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
# or use your own weights:
model = DEGAN(bin_weights='path/to/binary/weights.h5', en_weights='path/to/deblur/weights.h5', rn_weights='path/to/unwatermark/weights.h5')

binarized_image = model.binarize(image)
deblurred_image = model.deblur(image)
removed_noise_image = model.unwatermark(image)

# you can also instantiate only certain models:
model = DEGAN(en_weights=None, rn_weights=None)
```

For training custom weights, please refer to the [original repository](https://github.com/dali92002/DE-GAN).

### CLI

**Note: for using as a CLI, consider installing via `pipx`**

- Default weights: `degan binarize image.png`
- Custom weights: `degan binarize image.png --bin_weights path/to/binary/weights.h5`
- Other subcommands: `degan deblur`, `degan unwatermark`
- All options: `degan --help`

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

```

```
