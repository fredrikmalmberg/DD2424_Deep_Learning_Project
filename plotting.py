import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.color import lab2rgb, rgb2lab

import Network_Layers
import data_manager as data_manager
import frechet_inception_difference as fid


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
            self.plot(self.settings, self.model, self.w, self.img)

    def plot(self, settings, model, w, image_path, logs=None):
        predictModel = Network_Layers.create_model(settings, w, training=False)
        predictModel.set_weights(model.get_weights())

        cs = np.load('dataset/data/color_space.npy')
        picture = cv2.imread(image_path)
        img = rgb2lab(picture)
        L = img[:, :, :1]

        out = predictModel.predict(
            np.array(([L])), batch_size=1, verbose=1)
        img_predict = out[0, :, :, :]

        A = cs[np.argmax(img_predict, axis=2)][:, :, 0]
        B = cs[np.argmax(img_predict, axis=2)][:, :, 1]
        L = np.clip(L[:, :, 0], a_min=0, a_max=100)

        diff1 = A.shape[0] - L.shape[0]
        diff2 = A.shape[1] - L.shape[1]
        A = np.clip(A[int(diff1 // 2):-int(diff1 - diff1 // 2), int(diff2 // 2):-int(diff2 - diff2 // 2)], a_min=-110,
                    a_max=110)
        B = np.clip(B[int(diff1 // 2):-int(diff1 - diff1 // 2), int(diff2 // 2):-int(diff2 - diff2 // 2)], a_min=-110,
                    a_max=110)
        dpi = 80
        img_combined = np.swapaxes(np.array(([L, A, B])), 2, 0)  # why do I have to invert A and B
        picture = lab2rgb(img_combined.astype(np.float64))
        rotated_img = np.flip(ndimage.rotate(picture, -90), axis=1)

        f = plt.figure(figsize=(10, 20))
        ax1 = f.add_subplot(121)
        imgplot = plt.imshow((rotated_img * 255).astype(np.uint8))
        plt.title("Combined prediction")

        L = np.ones(A.shape) * 50
        img_combined = np.swapaxes(np.array(([L, A, B])), 2, 0)  # why do I have to invert A and B
        picture = lab2rgb(img_combined)
        rotated_img = np.flip(ndimage.rotate(picture, -90), axis=1)

        ax1 = f.add_subplot(122)
        imgplot = plt.imshow((rotated_img * 255).astype(np.uint8), interpolation='none')
        plt.title("Only color")
        plt.show()
        del predictModel
        return self


def plot_output(model, img_lab, img_AB):
    cs = np.load('dataset/data/color_space.npy')
    out = model.predict(
        np.array(([img_lab])), batch_size=20, verbose=1, steps=None, callbacks=None, max_queue_size=10,
        workers=1, use_multiprocessing=False
    )

    img_predict = out[0, :, :, :]
    # img_L = img_lab
    A = cs[np.argmax(img_predict, axis=2)][:, :, 0].T
    B = cs[np.argmax(img_predict, axis=2)][:, :, 1].T
    L = cv2.resize(img_lab, (64, 64))

    f = plt.figure(figsize=(20, 6))
    ax = f.add_subplot(141)
    img_combined = np.swapaxes(np.array(([L, -A.T, -B.T])), 2, 0)  # why do I have to invert A and B
    rotated_img = np.flip(ndimage.rotate(lab2rgb(img_combined), -90), axis=1)
    imgplot = plt.imshow((rotated_img * 255).astype(np.uint8))
    plt.title("Combined result")

    # and just the colour tone
    plt.subplot(1, 4, 2)
    lum = np.ones(A.shape) * 50
    img_combined = np.swapaxes(np.array(([lum, -A.T, -B.T])), 2, 0)  # why do I have to invert A and B
    rotated_img = np.flip(ndimage.rotate(lab2rgb(img_combined), -90), axis=1)
    imgplot = plt.imshow((rotated_img * 255).astype(np.uint8))
    plt.title("Color tone")

    plt.subplot(1, 4, 3)
    grid = np.ones((22, 22))
    gamut = np.sum(np.sum(img_AB, axis=0), axis=0)
    for i in range(-110, 110, 10):
        for j in range(-110, 110, 10):
            if np.any(np.all(np.array(([i, j])) == cs, axis=1)):
                for idx, c in enumerate(cs):
                    if np.all(np.array(([i, j])) == c):
                        grid[11 + int(i / 10), int(11 + j / 10)] += gamut[idx]
            else:
                grid[11 + int(i / 10), int(11 + j / 10)] = -300

    plt.imshow(grid, cmap='inferno')
    plt.yticks(range(22), range(110, -110, -10))
    plt.xticks(range(22), range(-110, 110, 10))
    plt.title("Original Gamut")

    plt.subplot(1, 4, 4)
    grid = np.ones((22, 22))
    gamut = np.sum(np.sum(img_predict, axis=0), axis=0)
    for i in range(-110, 110, 10):
        for j in range(-110, 110, 10):
            if np.any(np.all(np.array(([i, j])) == cs, axis=1)):
                for idx, c in enumerate(cs):
                    if np.all(np.array(([i, j])) == c):
                        grid[11 + int(i / 10), int(11 + j / 10)] += gamut[idx]
            else:
                grid[11 + int(i / 10), int(11 + j / 10)] = -300
    plt.imshow(grid, cmap='inferno')
    plt.yticks(range(22), range(110, -110, -10))
    plt.xticks(range(22), range(-110, 110, 10))
    plt.title("Predicted Gamut")

    # plt.savefig('demo.png', bbox_inches='tight') # Lucas needs this to compile
    plt.show()
