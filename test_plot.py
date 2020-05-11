import data_manager as data_manager
import frechet_inception_difference as fid

def colorize_benchmark_images(model,show_fid = True):
    data = data_manager.get_benchmark_images()
    if show_fid:
        originals = fid.return_benchmark_originals()
    for i in range(data['input'].shape[0]):
        if show_fid:     
            colorized = fid.return_colorized(model, data['input'][i][:, :, :])
            fid_val = fid.return_fid(colorized, originals[i])
            print("The Frechet Inception Difference is:",fid_val)
        plot_output(model, data['input'][i][:, :, :], data['target'][i][:, :, :])


def plot_output(model, img_lab, img_AB):
    from skimage.color import lab2rgb, rgb2lab
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from scipy import ndimage
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

    plt.savefig('demo.png', bbox_inches='tight') # Lucas needs this to compile
    # plt.show()