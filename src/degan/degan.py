from pathlib import Path

import cv2
import numpy as np

from .models import generator_model
from .utils import merge_image2, split2

# this works for editable, local, and wheel version, importlib is a mess for
# supporting all of them
THIS_DIR = Path(__file__).parent
WEIGHTS_DIR = THIS_DIR / "weights"

BIN_FNAME = WEIGHTS_DIR / "binarization_generator_weights.h5"
DEB_FNAME = WEIGHTS_DIR / "deblur_weights.h5"
WAT_FNAME = WEIGHTS_DIR / "watermark_rem_weights.h5"


class DEGAN:
    def __init__(
        self,
        bin_weights=BIN_FNAME,
        deb_weights=DEB_FNAME,
        wat_weights=WAT_FNAME,
        input_size=(256, 256, 1),
    ):
        self.input_size = input_size
        self.bin_weights = bin_weights
        self.deb_weights = deb_weights
        self.wat_weights = wat_weights
        self.binarizer = None
        self.deblurrer = None
        self.unwatermarker = None

    def load_weights(self):
        if self.bin_weights is None:
            self.binarizer = None
        else:
            self.binarizer = self._model_instantiate(
                self.bin_weights, biggest_layer=1024
            )

        if self.deb_weights is None:
            self.deblurrer = None
        else:
            self.deblurrer = self._model_instantiate(
                self.deb_weights, biggest_layer=1024
            )

        if self.wat_weights is None:
            self.unwatermarker = None
        else:
            self.unwatermarker = self._model_instantiate(
                self.wat_weights, biggest_layer=512
            )

    def _model_instantiate(self, model_weights, biggest_layer):
        model = generator_model(biggest_layer=biggest_layer)
        model.load_weights(model_weights)
        return model

    def _preprocess(self, in_image):
        h = ((in_image.shape[0] // 256) + 1) * 256
        w = ((in_image.shape[1] // 256) + 1) * 256

        test_padding = np.zeros((h, w)) + 1
        test_padding[: in_image.shape[0], : in_image.shape[1]] = in_image

        in_image_p = split2(test_padding.reshape(1, h, w, 1), 1, h, w)
        return in_image_p, (h, w)

    def _predict(self, in_image_p):
        predicted_list = []
        for pred in range(in_image_p.shape[0]):
            # FIX: the next line takes a lot of memory
            predicted_list.append(
                self._generator.predict(
                    in_image_p[pred].reshape(1, 256, 256, 1), verbose=0
                )
            )
        predicted_image = np.array(predicted_list)  # .reshape()
        return predicted_image

    def _postprocess(self, predicted_image, in_shape, reshaped):
        predicted_image = merge_image2(predicted_image, *reshaped)

        predicted_image = predicted_image[: in_shape[0], : in_shape[1]]
        predicted_image = predicted_image.reshape(
            predicted_image.shape[0], predicted_image.shape[1]
        )
        return predicted_image

    def _generate(self, image):
        image_, reshaped = self._preprocess(image)
        image_ = self._predict(image_)
        return self._postprocess(image_, image.shape, reshaped)

    def binarize(self, img):
        if self.binarizer is None:
            self.binarizer = self._model_instantiate(
                self.bin_weights, biggest_layer=1024
            )
        self._generator = self.binarizer
        img = self._generate(img)
        img = (img > 0.5).astype(np.float32)
        return img

    def deblur(self, img):
        if self.deblurrer is None:
            self.deblurrer = self._model_instantiate(
                self.deb_weights, biggest_layer=1024
            )
        self._generator = self.deblurrer
        img = self._generate(img)
        return img

    def unwatermark(self, img):
        if self.unwatermarker is None:
            self.unwatermarker = self._model_instantiate(
                self.wat_weights, biggest_layer=512
            )
        self._generator = self.unwatermarker
        img = self._generate(img)
        return img


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    return image.astype(np.float32)


def write_image(image, image_path):
    image_path = str(image_path)
    cv2.imwrite(image_path, (image * 255).astype(np.uint8))
