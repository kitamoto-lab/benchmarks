import numpy as np

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        max_shift = 0
        top = np.random.randint((h - new_h) / 2 - max_shift, (h - new_h) / 2 + max_shift + 1 )
        left = np.random.randint((w - new_w) / 2 - max_shift, (w - new_w) / 2 + max_shift + 1)
        image = image[top: top + new_h,
                      left: left + new_w]

        return image