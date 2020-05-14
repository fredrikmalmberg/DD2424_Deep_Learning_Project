import os
import re
from datetime import datetime

import cv2
import keras.models
import numpy as np
from skimage.color import rgb2lab
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from get_color_scale import generate_unique_color_space
from model import create_model
from dataobjects import settings


def assert_data_is_setup(settings):
    # Checks that we have training, validation and test folders. We assume if the folders exists that there is files
    # in them. Checking if files exists in a directory if the data set is big takes unnecessary time.
    if not os.path.isdir(settings.data_directory + "/train"):
        raise FileNotFoundError("No training data set found, please check that you the files and then run again")
    if not os.path.isdir(settings.data_directory + "/validation"):
        raise FileNotFoundError("No validation data set found, please check that you the files and then run again")
    if not os.path.isdir(settings.data_directory + "/test"):
        raise FileNotFoundError("No test data set found, please check that you the files and then run again")

    # Check that color space file is setup
    if not os.path.isfile("dataset/data/color_space.npy"):
        print("color_space.npy is missing, will generate a new one from the training data")
        generate_unique_color_space()
        print("New color_space.npy was generated successfully")

    print("The data is setup correctly")


def get_soft_enconding_ab(target, uniques):
    a = np.ravel(target[:, :, 0])
    b = np.ravel(target[:, :, 1])
    ab = np.vstack((a, b)).T
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(uniques)
    distances, indices = nbrs.kneighbors(ab)
    z = gaussian_kernel(distances, sigma=5)
    y = np.zeros((ab.shape[0], uniques.shape[0]))
    index = np.arange(ab.shape[0]).reshape(-1, 1)
    y[index, indices] = z
    y = y.reshape(target.shape[0], target.shape[1], uniques.shape[0])
    return y


def get_onehot_encoding(target, uniques):
    a = np.ravel(target[:, :, 0])
    b = np.ravel(target[:, :, 1])
    ab = np.vstack((a, b)).T
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(uniques)
    distances, indices = nbrs.kneighbors(ab)
    y = np.zeros((ab.shape[0], uniques.shape[0]))
    index = np.arange(ab.shape[0]).reshape(-1, 1)
    y[index, indices] = 1
    y = y.reshape(target.shape[0], target.shape[1], uniques.shape[0])
    return y


def gaussian_kernel(distance, sigma):
    """
    Gaussian kernel
    :param distance: Distance between data
    :param sigma: Desired sigma
    :return: Kernel matrix
    """

    num = np.exp(-np.power(distance, 2) / (2 * np.power(sigma, 2)))
    denom = np.sum(num, axis=1).reshape(-1, 1)
    return num / denom


def save_model(model, name=None):
    """
    Saves the model to disk. If no name provided it will use the time the model was saved
    :param model: The model to save/store
    :param name: A name to identify the model
    :return: The saved models name
    """

    # Sets the name of the model
    if name is None:
        now = datetime.now()
        name = now.strftime("%Y_%m_%d_%H_%M")
    model.save("trained_models/" + name)
    print("Saved model: {model_name} to disk successfully".format(model_name=name))
    return name


def load_model_old(path_and_name):
    """
    Loads a model from disk with the given param name.
    :param path_and_name: The path and the name to the model to be loaded
    :return: A model
    """
    # from model import create_model.lo

    # model = keras.models.load_model(path_and_name, custom_objects={'loss_function': loss_function})
    model = keras.models.load_model(path_and_name, compile=False)
    print("Loaded model: {model_name} from disk successfully".format(model_name=path_and_name))
    return model

def load_model(name, settings, w):
    model = create_model(settings, w)
    model.load_weights(name)
    return model

def save_lab_figures(dataset):
    """
    Converts rgb files into lab format
    :param dataset: Folder where desired data will be converted
    :return: void
    """
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
    """
    Loads images in lab format
    :param dataset: name of the dataset folder
    :param batch_size: How many data points to load
    :return: batch_size data tuple with input and target
    """

    data = {}
    folders = os.listdir('dataset/data_lab/{}/'.format(dataset))
    n_images = 0

    for file in folders:
        if not file.endswith(".txt") and not file.endswith('.JPEG'):
            n_images += 1

    input = np.zeros((batch_size, 256, 256, 1))
    target = np.zeros((batch_size, 64, 64, 313))
    batch = 0
    print("\n LOADING DATA {}".format(dataset))
    if os.path.exists('dataset/data/color_space.npy'):
        print("Loaded one hot encoding color space")
        uniques = np.load('dataset/data/color_space.npy')
    else:
        uniques = None
    for image in tqdm(folders):
        if not image.endswith(".txt") and not image.endswith('.JPEG'):
            if batch < batch_size:
                picture = np.load('dataset/data_lab/{}/{}'.format(dataset, image))
                input[batch] = picture[:, :, :1]
                tmp = cv2.resize(picture[:, :, 1:], (64, 64))
                target[batch] = get_soft_enconding_ab(tmp, uniques)
                batch += 1
    data['input'] = input
    data['target'] = target

    return data
