import cv2
import numpy as np
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator


def create_model(settings):
    """
    Creates a model and compiles it with parameters from the settings file
    :param settings: Settings for the network
    :return: A compiled model ready to be trained
    """
    regulizer = settings.regularizer
    initilizer = settings.kernel_initializer
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=settings.input_shape))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv1_1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(2, 2),
                            padding='same', kernel_regularizer=regulizer, name='conv1_2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv2_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(2, 2),
                            padding='same', kernel_regularizer=regulizer, name='conv2_2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv3_1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv3_2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(2, 2),
                            padding='same', kernel_regularizer=regulizer, name='conv3_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv4_1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv4_2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv4_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv5_1', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv5_2', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv5_3', dilation_rate=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv6_1', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv6_2', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv6_3', dilation_rate=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv7_1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv7_2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv7_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv8_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv8_2'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initilizer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv8_3'))
    model.add(layers.BatchNormalization())
    model.add(
        layers.Conv2D(settings.nr_output_classes, (1, 1), activation='softmax', padding='same', name='pred'))

    # Sets final parameters and compiles network
    sgd = optimizers.Adam(learning_rate=settings.learning_rate)
    model.compile(loss=settings.loss_function, optimizer=sgd, metrics=["accuracy"])
    return model


def create_generators(settings):
    """
    Creates the generators that feeds the training with data.
    :param settings: Setting object with desired parameters to use
    :return: train_generator, validate_generator. One generator for each type of feed
    """
    target_size = (settings.input_shape[0], settings.input_shape[1])
    train_data_generator = ImageDataGenerator(rescale=1. / 255, preprocessing_function=labTransform)
    train_generator = train_data_generator.flow_from_directory(directory='dataset/data/train', target_size=target_size,
                                                               batch_size=settings.batch_size, class_mode=None)
    validate_data_generator = ImageDataGenerator(rescale=1. / 255, preprocessing_function=labTransform)
    validate_generator = validate_data_generator.flow_from_directory(directory='dataset/data/validation',
                                                                     target_size=target_size,
                                                                     batch_size=settings.batch_size,
                                                                     class_mode=None)
    return train_generator, validate_generator


def labTransform(image):
    """
    Transform the images to the desired input format and converts to LAB color scalre
    :param image: Image to be converted
    :return:
    """
    image = cv2.resize(image, (256, 256))   # TODO catherine said we need to wrap this preprocess function somehow
    lab_input = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
    Input = image[:, :, :1]
    target = image[:, :, 1:]
    return lab_input, target


def train_network(settings):
    model = create_model(settings)

    train_generator, validate_generator = create_generators(settings)

    model.fit_generator(generator=train_generator, steps_per_epoch=settings.training_steps_per_epoch,
                        epochs=settings.nr_epochs,
                        validation_data=validate_generator, validation_steps=settings.validation_steps_per_epoch)
    return model
