import cv2
import os
import numpy as np
import re
from tqdm import tqdm


def import_data(type):
    data = {}
    folders = os.listdir('dataset/data/{}/'.format(type))
    n_images = 0

    for subfolder in folders:
        entry = os.listdir('dataset/data/{}/{}'.format(type, subfolder))
        for image in entry:
            n_images +=1

    x = np.zeros((n_images, 256,256, 3))
    y = np.empty(n_images, dtype=object)
    batch = 0
    print("\n LOADING DATA {}".format(type))
    for subfolder in tqdm(folders):
        entry = os.listdir('dataset/data/{}/{}'.format(type, subfolder))
        for image in entry:
            picture = cv2.resize(cv2.imread('dataset/data/{}/{}/{}'.format(type, subfolder, image)), (256,256))
            y[batch] = (re.sub(r'_.*', "", image))
            x[batch] = picture/255
            batch += 1
    data['x'] = x
    data['y'] = y
    return data


def print_picture(picture):
    cv2.imshow('image', picture)
    # not sure why but it need this 2 following rows to work
    cv2.waitKey(0)
    cv2.destroyAllWindows()
