import cv2
import os
import numpy as np
import re
from tqdm import tqdm
from skimage.color import rgb2lab
from sklearn.neighbors import NearestNeighbors


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
    target = np.zeros((batch_size, 256, 256, 313))
    batch = 0
    print("\n LOADING DATA {}".format(dataset))
    for image in tqdm(folders):
        if not image.endswith(".txt"):
            if batch < batch_size:
                picture = np.load('dataset/data_lab/{}/{}'.format(dataset, image))
                input[batch] = picture[:, :, :1]
                target[batch] = onehot_enconding_ab(picture[:, :, 1:]) # 313 layers at the end
                # target[batch] = picture[:, :, 1:] ## 2 layers at the end
                batch += 1
    data['input'] = input
    data['target'] = target

    return data


def onehot_enconding_ab(target):
    a = np.ravel(target[:, :, 0])
    b = np.ravel(target[:, :, 1])
    if os.path.exists('dataset/data/color_space.npy'):
        uniques = np.load('dataset/data/color_space.npy')
        #print('\n Loaded unique values')
    else:
        bins = np.arange(-110, 110, 10)
        bin_a = np.copy(a)
        bin_b = np.copy(b)
        for i in range(len(bins) - 1):
            bin_a = np.where((a < bins[i + 1]) & (a >= bins[i]), bins[i], bin_a)
            bin_b = np.where((b < bins[i + 1]) & (b >= bins[i]), bins[i], bin_b)
        uniques = np.unique(
            np.sort(np.array(list(set(tuple(sorted([m, n])) for m, n in zip(bin_a, bin_b)))), axis=0), axis=0)
        # np.save('dataset/data/color_space.npy', uniques)
        print('\n Created unique values')
    #print('staking')
    ab = np.vstack((a, b)).T
    #print('getting neighbors')
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(uniques)
    distances, indices = nbrs.kneighbors(ab)

    y = np.zeros((ab.shape[0], uniques.shape[0]))
    index = np.arange(ab.shape[0]).reshape(-1, 1)
    z = gaussian_kernel(distances, sigma=5)
    y[index, indices] = z
    y = y.reshape(target.shape[0], target.shape[1], uniques.shape[0])  # should be n_images X 256 X 256 X 313
    return y


def gaussian_kernel(distance, sigma):
    num = np.exp(-np.power(distance, 2) / (2 * np.power(sigma, 2)))
    denom = np.sum(num, axis=1).reshape(-1, 1)
    return num / denom


def print_picture(picture):
    cv2.imshow('image', picture)
    # not sure why but it need this 2 following rows to work
    cv2.waitKey(0)
    cv2.destroyAllWindows()
