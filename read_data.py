import cv2
import csv
import os
import numpy as np
import re
from tqdm import tqdm


def import_data(type):
    x = list()
    y = list()
    data = {}
    y_dict = {}
    with open('dataset/LOC_synset_mapping.txt', newline='\n') as inputfile:
        for i, row in enumerate(csv.reader(inputfile)):
            label = str(row[0]).split(' ')[0]
            y_dict[label] = i

    folders = os.listdir('dataset/data/{}/'.format(type))
    print("\n LOADING DATA {}".format(type))
    for subfolder in tqdm(folders):
        entry = os.listdir('dataset/data/{}/{}'.format(type, subfolder))
        for image in entry:
            picture = cv2.resize(cv2.imread('dataset/data/{}/{}/{}'.format(type, subfolder, image)), (256, 256))
            label = (re.sub(r'_.*', "", image))
            x.append(picture.reshape(-1, 1))
            y_hot = np.zeros((len(y_dict), 1))
            y_hot[y_dict[label]] = 1
            y.append(y_hot)

    data['x'] = x
    data['y'] = y
    return data


def print_picture(picture):
    pic = picture.reshape((256, 256, 3))
    cv2.imshow('image', pic)
    # not sure why but it need this 2 following rows to work
    cv2.waitKey(0)
    cv2.destroyAllWindows()
