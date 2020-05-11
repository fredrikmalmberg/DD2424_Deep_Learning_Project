from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from pre_process_crop import load_and_crop_img
from keras.applications.inception_v3 import preprocess_input
import keras_preprocessing.image
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # This line allows the network to use the GPU VRAM uncapped. !!! NEED THIS LINE FOR NETWORK TO RUN !!!
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
        # tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[1], True)
        # tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[1], True)
        # tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[1], True)
        # tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)

from data_manager import onehot_enconding_ab


def create_model(settings, training = True):
    """
    Creates a model and compiles it with parameters from the settings file
    :param settings: Settings for the network
    :return: A compiled model ready to be trained
    """
    # regulizer = settings.regularizer
    regulizer = None
    initializer = settings.kernel_initializer
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', kernel_regularizer=regulizer, name='conv1_1',
                            input_shape=settings.input_layer_shape))
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
                            padding='same', dilation_rate=1, kernel_regularizer=regulizer, name='conv4_1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=1, kernel_regularizer=regulizer, name='conv4_2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=1, kernel_regularizer=regulizer, name='conv4_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=2, kernel_regularizer=regulizer, name='conv5_1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=2, kernel_regularizer=regulizer, name='conv5_2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=2, kernel_regularizer=regulizer, name='conv5_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=2, kernel_regularizer=regulizer, name='conv6_1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=2, kernel_regularizer=regulizer, name='conv6_2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=2, kernel_regularizer=regulizer, name='conv6_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=1, kernel_regularizer=regulizer, name='conv7_1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=1, kernel_regularizer=regulizer, name='conv7_2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=1, kernel_regularizer=regulizer, name='conv7_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=1, kernel_regularizer=regulizer, name='conv8_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=1, kernel_regularizer=regulizer, name='conv8_2'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                            padding='same', dilation_rate=1, kernel_regularizer=regulizer, name='conv8_3'))
    # model.add(layers.BatchNormalization())
    #model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(settings.nr_colors_space, (1, 1), activation='softmax', kernel_initializer=initializer,
                            strides=(1, 1), padding='same', dilation_rate=1, name='pred',
                            ))
    from keras import backend as K

    from keras.activations import softmax

    if not training: # this is run when we are predicting
        model.add(layers.UpSampling2D((4, 4)))

    #print(model.summary())
    # Sets final parameters and compiles network
    sgd = tfa.optimizers.AdamW(learning_rate=settings.learning_rate, weight_decay=1e-3, beta_1=0.9, beta_2=0.99,
                               epsilon=1e-07)

    if settings.from_checkpoint:
        model.load_weights(load_checkpoint(model))
        print('successfully loaded checkpoint')

    def loss_temp(y_true, y_pred):
        y_pred_log = K.log(y_pred + K.epsilon())
        ret = -K.sum(tf.multiply(y_true, y_pred_log))
        return ret




    model.compile(loss=loss_temp, optimizer=sgd, metrics=["accuracy"])
    return model


def create_generator(settings, data_set):
    """
    Creates the generators that feeds the training with data.
    :param settings: Setting object with desired parameters to use
    :param data_set: The chossen data set to generate from (options train, validation, test)
    :return: A generator that generates data from the data_set
    """
    keras_preprocessing.image.iterator.load_img = load_and_crop_img
    if not (data_set == "train" or data_set == "validation" or data_set == "test"):
        raise NotImplementedError("Input data_set does not match allowed options (train, validation, test)")

    unique_colors = np.load('dataset/data/color_space.npy')  # The list of unique color combinations
    target_size = (settings.input_shape[0], settings.input_shape[1])  # Target the size the images will be resize to
    generator = ImageDataGenerator().flow_from_directory(directory=settings.data_directory + data_set,
                                                         target_size=target_size,
                                                         batch_size=settings.batch_size, class_mode=None)
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

        target_batch = cv2.resize(images_lab[:, :, 1:], (settings.output_shape[0], settings.output_shape[1]), cv2.INTER_CUBIC)
        targets[batch] = onehot_enconding_ab(target_batch, unique_colors)

        test = np.argmax(targets[0], axis=2)

        # TODO: RE WEIGHT HERE

        if settings.use_reweighting:
            lamda = 0.5
            q = targets[batch].shape[2]
            reweighted = (1 - lamda) * targets[batch] + lamda / q

            targets[batch] = np.power(reweighted, -1)

        input_batch = cv2.resize(images_lab[:, :, :1], (settings.input_shape[0], settings.input_shape[1]), cv2.INTER_CUBIC)
        inputs[batch] = input_batch.reshape((256,256,1))



        # START FM add to see some images during preprocess
        if (np.random.uniform() < 0.2) & False:
            print(images[batch].shape)
            from skimage.color import lab2rgb
            from scipy import ndimage
            import matplotlib.pyplot as plt
            L = images_lab[:, :, 0]
            A = images_lab[:, :, 1]
            B = images_lab[:, :, 2]
            img_combined = np.swapaxes(np.array(([L, A, B])), 2, 0)  # why do I have to invert A and B
            picture = lab2rgb(img_combined)
            rotated_img = np.flip(ndimage.rotate(picture, -90), axis=1)
            f = plt.figure(figsize=(40, 12))
            ax1 = f.add_subplot(131)
            _ = plt.imshow((rotated_img * 255).astype(np.uint8))
            plt.title("Combined input image in pre_process")
            #plt.show()

            L = cv2.resize(images_lab[:, :, 0], (settings.output_shape[0], settings.output_shape[1]))


            A = target_batch[:, :, 0]
            B = target_batch[:, :, 1]
            img_combined = np.swapaxes(np.array(([L, A, B])), 2, 0)  # why do I have to invert A and B
            picture = lab2rgb(img_combined)
            rotated_img = np.flip(ndimage.rotate(picture, -90), axis=1)
            ax1 = f.add_subplot(132)
            _ = plt.imshow((rotated_img * 255).astype(np.uint8))
            plt.title("Combined target image with one_hot")
            #plt.show()

            A = unique_colors[np.argmax(onehot_enconding_ab(target_batch, unique_colors), axis=2)][:, :, 0]
            B = unique_colors[np.argmax(onehot_enconding_ab(target_batch, unique_colors), axis=2)][:, :, 1]
            img_combined = np.swapaxes(np.array(([L, A, B])), 2, 0)  # why do I have to invert A and B
            picture = lab2rgb(img_combined)
            rotated_img = np.flip(ndimage.rotate(picture, -90), axis=1)
            ax1 = f.add_subplot(133)
            _ = plt.imshow((rotated_img * 255).astype(np.uint8))
            plt.title("Combined input and target image in pre_process")
            plt.show()
            # END FM add

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

    model = create_model(settings)
    train_generator = create_generator(settings, "train")
    validate_generator = create_generator(settings, "validation")
    settings.print_training_settings()
    # checkpoint = ModelCheckpoint('checkpoints/best_weights', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    # reduced_learning_rate = ReduceLROnPlateau('val_loss', factor=settings.learning_rate_reduction,
    #                                           patience=settings.patience, min_lr=settings.min_learning_rate, verbose=1)
    # callbacks_list = [checkpoint, reduced_learning_rate]
    print("Starting to train the network")
    start_time = datetime.now()
    model.fit(x=train_generator, epochs=settings.nr_epochs, steps_per_epoch=settings.training_steps_per_epoch,
              validation_data=validate_generator, validation_steps=settings.validation_steps_per_epoch,
              class_weight=class_weight)  # , callbacks=callbacks_list)
    execution_time = datetime.now() - start_time
    print("Training done. Execution time for the training was: ", execution_time)
    return model


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
