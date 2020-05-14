import os

import numpy as np
from tqdm import tqdm

import data_manager
import plotting


def create_rebalance_file():
    """
    Cretes a new file that stores the weights of the bins
    :return: 313x1 numpy array
    """

    folder_path = "dataset/dogs/train/"
    unique_colors = np.load('dataset/data/color_space.npy')
    folders = os.listdir(folder_path)
    bins = np.zeros((unique_colors.shape[0]))
    for subfolder in tqdm(folders):
        entry = os.listdir(folder_path + subfolder)
        for image in entry:
            image_path = f'{folder_path}{subfolder}/{image}'
            rgb_image = plotting.get_rgb_from_path(image_path)
            L, A, B = plotting.get_lab_channels_from_rgb(rgb_image)
            bins_image = data_manager.get_onehot_encoding(np.array(([A, B])).T, unique_colors)
            bins += np.sum(np.sum(bins_image, axis=0), axis=0)

    bins = bins / np.sum(bins)
    np.save("trained_models/prior_dog.npy", bins)


def get_re_weights(priors, lamb):
    q = priors.shape[0]
    w = (1 - lamb) * priors + lamb / q
    w = np.power(w, -1)
    w = w / np.sum(np.multiply(priors, w))
    w = w.astype(np.float32)
    return w
