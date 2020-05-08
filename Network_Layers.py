import cv2
import numpy as np
import tensorflow as tf
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # This line allows the network to use the GPU VRAM uncapped. !!! NEED THIS LINE FOR NETWORK TO RUN !!!
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    except RuntimeError as e:
        print(e)

from data_manager import onehot_enconding_ab


def create_model(settings):
    """
    Creates a model and compiles it with parameters from the settings file
    :param settings: Settings for the network
    :return: A compiled model ready to be trained
    """
    regulizer = settings.regularizer
    initializer = settings.kernel_initializer
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=settings.input_shape))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv1_1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, strides=(2, 2),
                            padding='same', kernel_regularizer=regulizer, name='conv1_2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv2_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(2, 2),
                            padding='same', kernel_regularizer=regulizer, name='conv2_2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv3_1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv3_2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(2, 2),
                            padding='same', kernel_regularizer=regulizer, name='conv3_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv4_1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv4_2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv4_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv5_1', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv5_2', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv5_3', dilation_rate=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv6_1', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv6_2', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv6_3', dilation_rate=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv7_1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv7_2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv7_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv8_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv8_2'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv8_3'))
    model.add(layers.BatchNormalization())
    model.add(
        layers.Conv2D(settings.nr_colors_space, (1, 1), activation='softmax', padding='same', name='pred'))

    # Sets final parameters and compiles network
    sgd = optimizers.Adam(learning_rate=settings.learning_rate)
    model.compile(loss=settings.loss_function, optimizer=sgd, metrics=["accuracy"])
    return model


def create_generators(settings, unique_colors):
    """
    Creates the generators that feeds the training with data.
    :param unique_colors: The list of unique color combination created from the data
    :param settings: Setting object with desired parameters to use
    :return: train_generator, validate_generator. One generator for one set of data
    """
    target_size = (settings.input_shape[0], settings.input_shape[1])
    train_generator = ImageDataGenerator().flow_from_directory(directory='dataset/data/train', target_size=target_size,
                                                               batch_size=settings.batch_size, class_mode=None)
    train_generator = generator_wrapper(train_generator, settings, unique_colors)

    validate_generator = ImageDataGenerator().flow_from_directory(directory='dataset/data/validation',
                                                                  target_size=target_size,
                                                                  batch_size=settings.batch_size,
                                                                  class_mode=None)

    validate_generator = generator_wrapper(validate_generator, settings, unique_colors)

    return train_generator, validate_generator


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
        images[batch] = images[batch] / 255.0           # Normalize data
        images_lab = rgb2lab(images[batch])             # Convert from rgb -> lab format

        target_batch = cv2.resize(images_lab[:, :, 1:], (settings.output_shape[0], settings.output_shape[1]))
        targets[batch] = onehot_enconding_ab(target_batch, unique_colors)

        input_batch = cv2.resize(images_lab[:, :, :], (settings.input_shape[0], settings.input_shape[1]))
        inputs[batch] = input_batch[:, :, :1]

    return inputs, targets


def train_network(settings, unique_colors):
    """
    Trains the network with settings gicen by the settings object and found classes in unique_colors
    :param settings:
    :param unique_colors:
    :return:
    """


    model = create_model(settings)

    train_generator, validate_generator = create_generators(settings, unique_colors)

    model.fit_generator(generator=train_generator, steps_per_epoch=settings.training_steps_per_epoch,
                        epochs=settings.nr_epochs,
                        validation_data=validate_generator, validation_steps=settings.validation_steps_per_epoch)
    return model
