import numpy as np
from tqdm import tqdm
import cv2
import os
from skimage.color import rgb2lab


def generate_unique_color_space():
    bins = np.arange(-110, 110, 10)
    target = np.zeros((10, 256, 256, 2))
    nr_image = 0
    folders = os.listdir('dataset/data/{}/'.format('train'))
    for subfolder in tqdm(folders):
        entry = os.listdir('dataset/data/{}/{}'.format('train', subfolder))
        for image in entry:
            if nr_image < target.shape[0]:
                picture_rgb = cv2.resize(cv2.imread('dataset/data/{}/{}/{}'.format('train', subfolder, image)),
                                         (256, 256)) / 255.0
                picture = rgb2lab(picture_rgb)
                if nr_image == 0:
                    a = np.ravel(picture[:, :, 1])
                    b = np.ravel(picture[:, :, 2])
                else:
                    a = np.append(a, np.ravel(picture[:, :, 1]))
                    b = np.append(b, np.ravel(picture[:, :, 2]))

                nr_image += 1

    bin_a = np.copy(a)
    bin_b = np.copy(b)
    for i in range(len(bins) - 1):
        bin_a = np.where((a < bins[i + 1]) & (a >= bins[i]), bins[i], bin_a)
        bin_b = np.where((b < bins[i + 1]) & (b >= bins[i]), bins[i], bin_b)
    uniques = np.unique(
        np.sort(np.array(list(set(tuple(sorted([m, n])) for m, n in zip(bin_a, bin_b)))), axis=0), axis=0)
    np.save('dataset/data/color_space.npy', uniques)


def main():
    generate_unique_color_space()


if __name__ == '__main__':
    main()
