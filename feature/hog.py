__author__ = 'fucus'

import sys
sys.path.append('../')

from skimage import color, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import skimage.io
from config import project


def get_1d_2d_hog(img):
    if len(img.shape) >= 3 and img.shape[2] == 3:
        img = color.rgb2gray(img)
    hog_image_1d, hog_image_2d = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
    return hog_image_1d, hog_image_2d


def get_hog(img):
    """
    :param img: the 2d rbg image, represented by numpy
    :return: list of feature
    """
    hog_image_1d, hog_image_2d = get_1d_2d_hog(img)
    hog = list(hog_image_1d)
    res = [int(x * 100) for x in hog]
    return res

if __name__ == '__main__':
    img = skimage.io.imread("%s/001.bmp" % project.test_data_path)
    hog_image_1d, hog_image_2d = get_1d_2d_hog(img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image_2d, in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()