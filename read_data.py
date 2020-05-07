import cv2
import os
import numpy as np
import re
from tqdm import tqdm
from skimage.color import rgb2lab


def save_lab_figures(dataset):
    folders = os.listdir('dataset/data/{}/'.format(dataset))
    for subfolder in tqdm(folders):
        entry = os.listdir('dataset/data/{}/{}'.format(dataset, subfolder))
        for image in entry:
            picture_rgb = cv2.resize(cv2.imread('dataset/data/{}/{}/{}'.format(dataset, subfolder, image)),
                                     (256, 256)) / 255.0
            picture = rgb2lab(picture_rgb)
            pic_name = (re.sub(r"\..*", "", image))
            np.save('dataset/data_lab/{}/{}'.format(dataset, pic_name), picture)


def import_data(dataset, batch_size):
    data = {}
    folders = os.listdir('dataset/data_lab/{}/'.format(dataset))
    n_images = 0

    for file in folders:
        if not file.endswith(".txt"):
            n_images += 1

    input = np.zeros((batch_size, 256, 256, 1))
    target = np.zeros((batch_size, 64, 64, 2))
    batch = 0
    print("\n LOADING DATA {}".format(dataset))
    for image in tqdm(folders):
        if not image.endswith(".txt"):
            while batch < batch_size:
                picture = np.load('dataset/data_lab/{}/{}'.format(dataset, image))
                input[batch] = picture[:, :, :1]
                target[batch] = cv2.resize(picture[:, :, 1:], (64, 64))
                batch += 1
    data['input'] = input
    data['target'] = target
    return data


def print_picture(picture):
    cv2.imshow('image', picture)
    # not sure why but it need this 2 following rows to work
    cv2.waitKey(0)
    cv2.destroyAllWindows()
