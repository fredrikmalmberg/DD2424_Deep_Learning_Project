import os

import cv2
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from scipy import ndimage
from scipy.linalg import sqrtm
from skimage.color import lab2rgb
from skimage.transform import resize

# code adapted from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

def return_colorized(model, img_lab):
    cs = np.load('dataset/data/color_space.npy')
    out = model.predict(np.array(([img_lab])), batch_size=20, verbose=1, steps=None, callbacks=None, max_queue_size=10,
                        workers=1, use_multiprocessing=False)
    img_predict = out[0, :, :, :]
    A = cs[np.argmax(img_predict, axis=2)][:, :, 0].T
    B = cs[np.argmax(img_predict, axis=2)][:, :, 1].T
    L = cv2.resize(img_lab, (64, 64))
    img_combined = np.swapaxes(np.array(([L, -A.T, -B.T])), 2, 0)  # why do I have to invert A and B
    rotated_img = (np.flip(ndimage.rotate(lab2rgb(img_combined), -90), axis=1)) * 255
    return rotated_img


def return_benchmark_originals():
    dataset = 'benchmark_images'
    folders = os.listdir('dataset/data_lab/{}/'.format(dataset))
    images = []
    for image in folders:
        if not image.endswith(".txt") and not image.endswith('.JPEG'):
            picture = np.load('dataset/data_lab/{}/{}'.format(dataset, image))
            images.append(picture)
    return images


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)


def calculate_fid(model, colorized_imgs, original_imgs):
    activities1 = model.predict(colorized_imgs)
    activities2 = model.predict(original_imgs)
    mu1, sigma1 = activities1.mean(axis=0), np.cov(activities1, rowvar=False)
    mu2, sigma2 = activities2.mean(axis=0), np.cov(activities2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def return_fid(colorized_imgs, original_imgs):
    """
    Returns the Frechet inception distance (fid) between two images. The pictures do not need to be the same size.
    :param colorized_imgs: The colorized image by the network
    :param original_imgs: The original image
    :return:
    """

    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    inputs = scale_images(colorized_imgs, (299, 299, 3))
    targets = scale_images(original_imgs, (299, 299, 3))
    fid = calculate_fid(model, inputs, targets)
    return fid
