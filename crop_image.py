import random
import keras_preprocessing.image
import numpy as np


def crop_img_patch(path, grayscale=False, color_mode='rgb', target_size=None,
                   interpolation='nearest'):
    # This function picks out a patch of size target_size. If the image is too small it resamples it so the smallest
    # dimension fits the target and then crops it. Since this is part of the generator it works as a wrapper of the
    # keras_preprocessing.image.utils.load_img function below and adds the cropping/resizing on top of that

    img = keras_preprocessing.image.utils.load_img(path,
                                                   grayscale=grayscale,
                                                   color_mode=color_mode,
                                                   target_size=None,
                                                   interpolation=interpolation)

    target_w = target_size[1]
    target_h = target_size[0]
    input_w, input_h = img.size

    resample = keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

    # We need to check if the image is smaller than the target...
    if (target_w >= input_w) or (target_h >= input_h):
        ratio = np.max([(target_w)/input_w, (target_h)/input_h])
        ratios = int(input_w * ratio), int(input_h * ratio)
        img = img.resize(ratios, resample=resample)
        input_w, input_h = img.size

    shift_x = 0
    shift_y = 0

    if int((input_w - target_w)) > 0:
        shift_x = random.randint(0, int((input_w - target_w)))
    if int((input_h - target_h)) > 0:
        shift_y = random.randint(0, int((input_h - target_h)))

    return img.crop((shift_x, shift_y, target_w + shift_x, target_h + shift_y))

