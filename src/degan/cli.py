from pathlib import Path

from .degan import DEGAN, load_image, write_image


class Main:
    def __init__(
        self,
        out_dir=None,
        bin_weights=None,
        deb_weights=None,
        wat_weights=None,
    ):
        if out_dir is None:
            out_dir = Path(".")
        self._out_dir = out_dir

        degan_kwargs = {}
        if bin_weights is not None:
            degan_kwargs["bin_weights"] = bin_weights
        if deb_weights is not None:
            degan_kwargs["deb_weights"] = deb_weights
        if wat_weights is not None:
            degan_kwargs["wat_weights"] = wat_weights
        self._degan = DEGAN(**degan_kwargs)

    def __load(self, *in_images):
        if not hasattr(self, "in_images"):
            self.in_images = []
            self.in_paths = in_images
            for img in in_images:
                self.in_images.append(load_image(img))

    def binarize(self, *in_images):
        """Binarize images"""
        self.__load(*in_images)
        for i, img in enumerate(self.in_images):
            img = self._degan.binarize(img)
            new_name = Path(self.in_paths[i]).name.replace(".png", "_bin.png")
            write_image(img, self._out_dir / new_name)
            self.in_images[i] = img

    def deblur(self, *in_images):
        """Deblur images"""
        self.__load(*in_images)
        for i, img in enumerate(self.in_images):
            img = self._degan.deblur(img)
            new_name = Path(self.in_paths[i]).name.replace(".png", "_deb.png")
            write_image(img, self._out_dir / new_name)
            self.in_images[i] = img

    def unwatermark(self, *in_images):
        """Unwatermark images"""
        self.__load(*in_images)
        for i, img in enumerate(self.in_images):
            img = self._degan.unwatermark(img)
            new_name = Path(self.in_paths[i]).name.replace(".png", "_wat.png")
            write_image(img, self._out_dir / new_name)
            self.in_images[i] = img


def main():
    import fire

    fire.Fire(Main)


if __name__ == "__main__":
    main()
