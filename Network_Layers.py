import os
from datetime import datetime

import cv2
import keras_preprocessing.image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import lab2rgb
from skimage.color import rgb2lab

from crop_image import crop_img_patch
from model import create_model
from plotting import combine_lab, epoch_plot

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # This line allows the network to use the GPU VRAM uncapped. !!! NEED THIS LINE FOR NETWORK TO RUN !!!
        for idx, g in enumerate(gpus):
            tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[idx], True)
        # tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)

from data_manager import get_soft_enconding_ab


def create_generator(settings, data_set):
    """
    Creates the generators that feeds the training with data.
    :param settings: Setting object with desired parameters to use
    :param data_set: The chossen data set to generate from (options train, validation, test)
    :return: A generator that generates data from the data_set
    """
    keras_preprocessing.image.iterator.load_img = crop_img_patch
    if not (data_set == "train" or data_set == "validation" or data_set == "test"):
        raise NotImplementedError("Input data_set does not match allowed options (train, validation, test)")

    unique_colors = np.load('dataset/data/color_space.npy')  # The list of unique color combinations
    target_size = (settings.input_shape[0], settings.input_shape[1])  # Target the size the images will be resize to
    generator = ImageDataGenerator().flow_from_directory(directory=settings.data_directory + data_set,
                                                         target_size=target_size,
                                                         batch_size=settings.batch_size, class_mode=None,
                                                         interpolation='lanczos', )
    return generator_wrapper(generator, settings, unique_colors)


def generator_wrapper(generator, settings, unique_colors):
    """
    Wrapper function is used so we can have a pre_process function that use multiple inputs compared to the
    standard implementation from keras that only take the image batch as input.
    :param generator: The generator that will be used to generate data
    :param settings: Settings objective that sets parameters
    :param unique_colors: Unique color combination created by the data
    :return: a batch of input and target
    """
    while True:
        x = generator.next()
        inputs, targets = pre_process(x, settings, unique_colors)
        yield inputs, targets


def pre_process(images, settings, unique_colors):
    """
    Transform the images to the desired input format and converts to LAB color format
    :param unique_colors:
    :param settings:
    :param images: Images to be converted
    :return: Pre processed data batch
    """
    inputs = np.zeros((images.shape[0], settings.input_shape[0], settings.input_shape[1], settings.input_shape[2]))
    targets = np.zeros((images.shape[0], settings.output_shape[0], settings.output_shape[1], settings.output_shape[2]))
    for batch in range(images.shape[0]):
        images[batch] = images[batch] / 255.0  # Normalize data
        images_lab = rgb2lab(images[batch])  # Convert from rgb -> lab format
        target_batch = cv2.resize(images_lab[:, :, 1:], (settings.output_shape[0], settings.output_shape[1]))
        targets[batch] = get_soft_enconding_ab(target_batch, unique_colors)
        inputs[batch] = images_lab[:, :, :1] - 50

    # Used for debugging
    if (np.random.uniform() < 0.1) & settings.plot_random_imgs_from_generator:
        L = images_lab[:, :, :1]
        print(np.min(L))
        L_sized = cv2.resize(L, (settings.output_shape[0], settings.output_shape[1]))
        print(L_sized.shape)
        A = unique_colors[np.argmax(targets[-1], axis=2)][:, :, 0]
        B = unique_colors[np.argmax(targets[-1], axis=2)][:, :, 1]
        lab_image = combine_lab(L_sized, A, B)
        rgb_image = lab2rgb(lab_image)
        f = plt.figure(figsize=(10, 20))
        ax1 = f.add_subplot(121)
        imgplot = plt.imshow(rgb_image)
        plt.title("L Input and target A B")
        ax1 = f.add_subplot(122)
        print(L.shape)
        L = cv2.resize(L, (settings.input_shape[0], settings.input_shape[1]))
        print(L.shape)
        A = np.zeros(L.shape)
        B = np.zeros(L.shape)
        lab_image = combine_lab(L, A, B)
        rgb_image = lab2rgb(lab_image)
        imgplot = plt.imshow(rgb_image)
        plt.title("L Input")
        plt.show()
    return inputs, targets


def load_checkpoint(model):
    """
    Load the best checkpoint from the most recent training
    :param: model
    :return: weight file from checkpoint
    """
    path = 'checkpoints/'
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    checkpoint = max(paths, key=os.path.getctime)
    return checkpoint


def train_network(settings, class_weight=None):
    """
    Trains the network with settings given by the settings object and found classes in unique_colors
    :param class_weight: The rebalancing of the classes
    :param settings: Settings object with chosen parameters
    :return: A trained model
    """

    model = create_model(settings, class_weight)
    train_generator = create_generator(settings, "train")
    validate_generator = create_generator(settings, "validation")
    settings.print_training_settings()
    callbacks_list = get_callback_functions(settings, model, class_weight)
    print("Starting to train the network")
    start_time = datetime.now()
    model.fit(x=train_generator, epochs=settings.nr_epochs, steps_per_epoch=settings.training_steps_per_epoch,
              validation_data=validate_generator, validation_steps=settings.validation_steps_per_epoch,
              callbacks=callbacks_list)
    execution_time = datetime.now() - start_time
    print("Training done. Execution time for the training was: ", execution_time)
    return model

def train_pretrained_model(model,settings, w):

    train_generator = create_generator(settings, "train")
    validate_generator = create_generator(settings, "validation")
    checkpoint = ModelCheckpoint('checkpoints/best_weights', monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    class_weight = w
    callbacks_list = get_callback_functions(settings, model, class_weight)
    print("Starting to train the pretrained network")
    model.fit(x=train_generator, epochs=settings.nr_epochs, steps_per_epoch=settings.training_steps_per_epoch,
              validation_data=validate_generator, class_weight=w, validation_steps=settings.validation_steps_per_epoch,
              callbacks=callbacks_list)
    return model


def get_callback_functions(settings, model, class_weight):
    """
    Returns callback functions used when training
    :param settings: Settings object with chosen parameters
    :param model: The model to be trained
    :param class_weight:
    :return: List of callback functions used when training
    """
    callbacks_list = []
    if settings.use_checkpoint:
        time_started = datetime.now().strftime("%Y_%m_%d_%H_%M")
        callbacks_list.append(ModelCheckpoint('checkpoints/' + time_started,
                                              monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))
    if settings.use_plotting:
        callbacks_list.append(
            epoch_plot(settings, model, class_weight,
                       'dataset/data/train_temp/n01440764/123.JPEG'))  # Todo make this dynamic
    if settings.use_reducing_lr:
        callbacks_list.append(ReduceLROnPlateau('val_loss', factor=settings.learning_rate_reduction,
                                                patience=settings.patience, min_lr=settings.min_learning_rate,
                                                verbose=1))
    if settings.use_loss_plotting:
        loss_logger = CSVLogger('log.csv', append=True, separator=';')
        callbacks_list.append(loss_logger)

    return callbacks_list


def evaluate_model(model, settings, verbose=1):
    """
    Evaluates the model on the stored test data set in the project
    Prints the result of the evaluation
    :param model: A trained model
    :param settings: Settings object with chosen parameters
    :param verbose: Handles the how much logs the evaluate model produce
    :return: void
    """

    test_generator = create_generator(settings, "test")
    print("Evaluating performance of model on test data set")
    scores = model.evaluate(x=test_generator, steps=settings.test_step_size, verbose=verbose)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def evaluate_model_on(model, settings, inputs, targets, verbose=1):
    """
    Evaluates the model on the param inputs and targets
    Prints the result of the evaluation
    :param targets: Data values
    :param inputs: Labels attached to the inputs
    :param model: A trained model
    :param settings: Settings object with chosen parameters
    :param verbose: Handles the how much logs the evaluate model produce
    :return: void
    """
    print("Evaluating performance of provided data set")
    scores = model.evaluate(x=inputs, y=targets, steps=settings.test_step_size, verbose=verbose)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
