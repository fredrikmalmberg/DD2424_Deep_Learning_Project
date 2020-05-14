import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import lab2rgb, rgb2lab
from model import create_model
import data_manager as data_manager
import frechet_inception_difference as fid
from dataobjects import settings


def colorize_benchmark_images(model, show_fid=True):
    data = data_manager.get_benchmark_images()
    if show_fid:
        originals = fid.return_benchmark_originals()
    for i in range(data['input'].shape[0]):
        if show_fid:
            colorized = fid.return_colorized(model, data['input'][i][:, :, :])
            fid_val = fid.return_fid(colorized, originals[i])
            print("The Frechet Inception Difference is:", fid_val)
        plot_output(model, data['input'][i][:, :, :], data['target'][i][:, :, :])
        plt.savefig('demo{}.png'.format(i), bbox_inches='tight')  # Lucas needs this to compile


class epoch_plot(keras.callbacks.Callback):
    def __init__(self, settings, model, w, image_path, ):
        self.settings = settings
        self.model = model
        self.w = w
        self.img = image_path

    def on_train_batch_begin(self, batch, logs=None):
        if batch % self.settings.training_steps_per_epoch == 0:
            plot_prediction(self.settings, self.model, self.w, self.img)


def plot_prediction(settings, model, w, image_path, savefig=False, T = 0.0):
    # This functions makes a prediction given an image path and plots it
    # Loading the color space bins
    cs = np.load('dataset/data/color_space.npy')

    # And the image
    rgb_image = get_rgb_from_path(image_path)
    # Getting LAB channels from rgb image
    L, A, B = get_lab_channels_from_rgb(rgb_image)

    # Reshaping input to fit model
    input_to_model = np.empty((1, L.shape[0], L.shape[1], 1))
    input_to_model[0, :, :, 0] = L - 50  # We train with shifted L so lets use it here as well

    # Creating a new model, setting weights and predicting
    predictModel = create_model(settings, w, training=False)
    predictModel.set_weights(model.get_weights())
    out = predictModel.predict(input_to_model, batch_size=1, verbose=1)

    # Removing the model
    del predictModel

    # Getting single image from prediction
    img_predict = out[0, :, :, :]

    # Picking out the A and B values from the predicted bins
    # If temperature is 0 we take mode otherwise annealed mean
    if T == 0.:
        A = cs[np.argmax(img_predict, axis=-1)][:, :, 0]
        B = cs[np.argmax(img_predict, axis=-1)][:, :, 1]
    else:
        # ugly debug stuff
        # z = img_predict[10, 10]
        # plt.plot(cs[:, 0], img_predict[10, 10])
        # plt.show()
        # z_temp = np.exp(np.log(z + epsilon) / T) / (np.sum(np.exp(np.log(z + epsilon) / T)) + epsilon)
        # plt.plot(cs[:, 0], z_temp)
        # plt.show()
        # print("predicted mode for A for pixel ", cs[np.argmax(z, axis=-1)][0])
        # print("predicted mean for A for pixel", np.sum(np.multiply(z, cs[:, 0])))
        # print("predicted annealed mean for A for pixel", np.sum(np.multiply(z_temp, cs[:, 0])))

        import sys
        epsilon = sys.float_info.epsilon
        A = np.zeros(img_predict[:,:,0].shape)
        B = np.zeros(img_predict[:,:,0].shape)
        for h in range(img_predict.shape[0]):
            for w in range(img_predict.shape[1]):
                z = img_predict[h,w]
                z_temp = np.exp(np.log(z + epsilon) / T) / (np.sum(np.exp(np.log(z + epsilon) / T)) + epsilon)
                A[h,w] = np.sum(np.multiply(z_temp, cs[:,0]))
                B[h,w] = np.sum(np.multiply(z_temp, cs[:,1]))

    # Some magic to match dimensions since we are losing a few pixels in the forward pass
    # When john fixes the model, we wont be needing it anymore :D
    diff1 = A.shape[0] - L.shape[0]
    diff2 = A.shape[1] - L.shape[1]

    crop1 = int(diff1 // 2)
    crop2 = -int(diff1 - diff1 // 2)
    if crop2 == 0:
        crop2 = A.shape[0] + 1
    crop3 = int(diff2 // 2)
    crop4 = -int(diff2 - diff2 // 2)
    if crop4 == 0:
        crop4 = A.shape[1] + 1

    # Cropping (and clipping is probably not needed if color space is correct)
    A = np.clip(A[crop1:crop2, crop3:crop4], a_min=-110, a_max=110)
    B = np.clip(B[crop1:crop2, crop3:crop4], a_min=-110, a_max=110)

    # Putting the image back together
    lab_image = combine_lab(L, A, B)

    # Back to rgb for plotting
    predicted_rgb_image = lab2rgb(lab_image)
    f = plt.figure(figsize=(10, 10))
    ax1 = f.add_subplot(131)
    predition = plt.imshow(predicted_rgb_image)
    plt.title("Combined prediction")

    # Just AB plotting
    lab_image = combine_lab_no_l_channel(A, B)
    predicted_rgb_image_ab = lab2rgb(lab_image)
    ax1 = f.add_subplot(132)
    predition_colors = plt.imshow(predicted_rgb_image_ab)
    plt.title("A B prediction")


    # Ground truth RGB
    ax1 = f.add_subplot(133)
    imgplot = plt.imshow(rgb_image)
    plt.title("Original Image")

    # plt.show()
    print("Predicted values for L, A & B: ")
    print("L: {} to {}.".format(np.min(L), np.max(L)))
    print("A: {} to {}.".format(np.min(A), np.max(A)))
    print("B: {} to {}.".format(np.min(B), np.max(B)))

    import re
    # fig_name = re.sub('\..*', '', image_path.rsplit('/',1)[1])
    fig_name = re.sub('\..*', '', image_path)

    if savefig:
        lucas = plt.figure()
        pred_ab = plt.imshow(predicted_rgb_image_ab)
        plt.savefig(str(fig_name) + '_pred_color.png')
        pred_img = plt.imshow(predicted_rgb_image)
        plt.savefig(str(fig_name) + '_pred.png')

    plt.show()
    return


def plot_unique_colours_gamut(unique_colors):
    # Plots the gamut that we are using ie the color of the bins
    grid = np.ones((22, 22))
    L = np.ones(grid.shape) * 50
    A = np.ones(grid.shape)
    B = np.ones(grid.shape)
    for i in range(-110, 110, 10):
        for j in range(-110, 110, 10):
            if np.any(np.all(np.array(([i, j])) == unique_colors, axis=1)):
                for idx, c in enumerate(unique_colors):
                    if np.all(np.array(([i, j])) == c):
                        A[11 + int(i / 10), int(11 + j / 10)] = unique_colors[idx][0]
                        B[11 + int(i / 10), int(11 + j / 10)] = unique_colors[idx][1]
            else:
                L[11 + int(i / 10), int(11 + j / 10)] = 0

    img_combined = combine_lab(L, A, B)
    picture = lab2rgb(img_combined)
    _ = plt.imshow(picture)
    plt.yticks(range(22), range(110, -110, -10))
    plt.xticks(range(22), range(-110, 110, 10))
    plt.title("Gamut for unique colours")
    plt.show()


def plot_weights_gamut(unique_colors, priors):
    # Plots the gamut that we are using ie the color of the bins
    grid = np.ones((22, 22))
    gamut = priors
    for i in range(-110, 110, 10):
        for j in range(-110, 110, 10):
            if np.any(np.all(np.array(([i, j])) == unique_colors, axis=1)):
                for idx, c in enumerate(unique_colors):
                    if np.all(np.array(([i, j])) == c):
                        grid[11 + int(i / 10), int(11 + j / 10)] += gamut[idx]
            else:
                grid[11 + int(i / 10), int(11 + j / 10)] = 0
    plt.imshow(grid, cmap='inferno')
    plt.yticks(range(22), range(110, -110, -10))
    plt.xticks(range(22), range(-110, 110, 10))
    plt.title("Original Gamut")
    plt.show()


def combine_lab(L, A, B):
    # Combines LAB to image
    img_combined = np.zeros(([A.shape[0], A.shape[1], 3]))
    img_combined[:, :, 0] = L
    img_combined[:, :, 1] = A
    img_combined[:, :, 2] = B
    return img_combined


def combine_lab_no_l_channel(A, B):
    # Combines A and B with a solid L to showcase colors only
    img_combined = np.zeros(([A.shape[0], A.shape[1], 3]))
    L = np.ones(A.shape) * 50
    img_combined[:, :, 0] = L
    img_combined[:, :, 1] = A
    img_combined[:, :, 2] = B
    return img_combined


def get_rgb_from_path(path):
    img = io.imread(path)
    return img


def get_rgb_from_lab(lab_image):
    return lab2rgb(lab_image)


def get_lab_channels_from_rgb(rgb_image):
    # Returns the LAB channels from RGB image
    images_lab = rgb2lab(rgb_image)
    L = images_lab[:, :, 0]
    A = images_lab[:, :, 1]
    B = images_lab[:, :, 2]
    return L, A, B


def plot_gamut_from_bins(img_AB, unique_colors):
    # Plots the gamut of an image
    grid = np.ones((22, 22))
    gamut = np.sum(np.sum(img_AB, axis=0), axis=0)
    for i in range(-110, 110, 10):
        for j in range(-110, 110, 10):
            if np.any(np.all(np.array(([i, j])) == unique_colors, axis=1)):
                for idx, c in enumerate(unique_colors):
                    if np.all(np.array(([i, j])) == c):
                        grid[11 + int(i / 10), int(11 + j / 10)] += gamut[idx]
            else:
                grid[11 + int(i / 10), int(11 + j / 10)] = 0
    plt.imshow(grid, cmap='inferno')
    plt.yticks(range(22), range(110, -110, -10))
    plt.xticks(range(22), range(-110, 110, 10))
    plt.title("Original Gamut")
    plt.show()


def plotting_demo():
    # This is just to show the different plotting functions
    # lets print the unique color bins
    unique_colors = np.load('dataset/data/color_space.npy')

    # first we try to read, split to lab, recombine and print
    image_path = 'dataset/data/train/n01440764/n01440764_141.JPEG'
    rgb_image = get_rgb_from_path(image_path)
    L, A, B = get_lab_channels_from_rgb(rgb_image)
    img_combined = combine_lab(L, A, B)
    picture = get_rgb_from_lab(img_combined)
    f = plt.figure(figsize=(10, 12))
    ax1 = f.add_subplot(131)
    _ = plt.imshow(picture)
    plt.title("Combined input image in pre_process")

    # Then lets try to recombine from one hot encoded bin values for A and B
    A = unique_colors[np.argmax(get_soft_enconding_ab(np.array(([A, B])).T, unique_colors), axis=2)][:, :, 0].T
    B = unique_colors[np.argmax(get_soft_enconding_ab(np.array(([A, B])).T, unique_colors), axis=2)][:, :, 1].T
    img_combined = combine_lab(L, A, B)
    picture = get_rgb_from_lab(img_combined)
    ax1 = f.add_subplot(132)
    _ = plt.imshow(picture)
    plt.title("Combined input image with one_hot")

    # Then just the AB channels
    A = unique_colors[np.argmax(get_soft_enconding_ab(np.array(([A, B])).T, unique_colors), axis=2)][:, :, 0].T
    B = unique_colors[np.argmax(get_soft_enconding_ab(np.array(([A, B])).T, unique_colors), axis=2)][:, :, 1].T
    img_combined = combine_lab_no_l_channel(A, B)
    picture = get_rgb_from_lab(img_combined)
    ax1 = f.add_subplot(133)
    _ = plt.imshow(picture)
    plt.title("Input image with one_hot")
    plt.show()

    img_AB = get_soft_enconding_ab(np.array(([A, B])).T, unique_colors)
    plot_gamut_from_bins(img_AB, unique_colors)

    # plot uniques
    plot_unique_colours_gamut(unique_colors)


def colorize_images_in_folder(settings, model, w, folder_path):
    images = os.listdir(folder_path)
    for image in tqdm(images):
        if 'pred' not in image:
            file_path = folder_path + image
            plot_prediction(settings, model, w, file_path, savefig=True)


def plot_epoch_metrics():
    metric_data = np.genfromtxt('log.csv', delimiter=';')
    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    plt.plot(metric_data[:, 0], metric_data[:, 1], label='Training Accuracy')
    plt.plot(metric_data[:, 0], metric_data[:, 3], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(122)
    plt.plot(metric_data[:, 0], metric_data[:, 2], label='Training Loss')
    plt.plot(metric_data[:, 0], metric_data[:, 4], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
