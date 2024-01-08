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
                print("Loading", img)
                self.in_images.append(load_image(img))

    def process_images(self, process_type, *in_images):
        """Process images according to the given process_type"""
        if process_type not in ["binarize", "deblur", "unwatermark"]:
            raise ValueError("Unsupported process type: {}".format(process_type))

        self.__load(*in_images)
        suffix_map = {
            "binarize": ".bin.png",
            "deblur": ".deb.png",
            "unwatermark": ".wat.png",
        }

        for i, img in enumerate(self.in_images):
            print(process_type.capitalize(), self.in_paths[i])

            # Dynamically call the appropriate method from _degan object
            process_method = getattr(self._degan, process_type)
            img = process_method(img)

            new_path = (
                Path(self.in_paths[i])
                .with_suffix(suffix_map[process_type])
                .relative_to(self._out_dir)
            )
            print("Writing", new_path)
            write_image(img, new_path)
            self.in_images[i] = img

    # The methods below now call the new generic process_images method with the appropriate process_type
    def binarize(self, *in_images):
        """Binarize images"""
        self.process_images("binarize", *in_images)

    def deblur(self, *in_images):
        """Deblur images"""
        self.process_images("deblur", *in_images)

    def unwatermark(self, *in_images):
        """Unwatermark images"""
        self.process_images("unwatermark", *in_images)


def main():
    import fire

    fire.Fire(Main)


if __name__ == "__main__":
    main()
