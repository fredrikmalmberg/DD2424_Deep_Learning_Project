from keras import losses, models, layers, optimizers, regularizers
import read_data
import cv2
import numpy as np
from skimage.color import rgb2lab
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator


def buildLayers(dimensions):
    """
    We use He initialization because it tends to give better results when using ReLu activation.
    :param dimensions: Dimension of the input data
    :return: A model
    """
    l2_reg = regularizers.l2()  # Todo find the value of the regularization in the paper, default here is 0.01

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=dimensions[1:]))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv1_1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(2, 2),
                            padding='same', kernel_regularizer=l2_reg, name='conv1_2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv2_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(2, 2),
                            padding='same', kernel_regularizer=l2_reg, name='conv2_2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv3_1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv3_2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(2, 2),
                            padding='same', kernel_regularizer=l2_reg, name='conv3_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv4_1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv4_2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv4_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv5_1', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv5_2', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv5_3', dilation_rate=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv6_1', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv6_2', dilation_rate=2))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv6_3', dilation_rate=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv7_1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv7_2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv7_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.UpSampling2D((8, 8)))    # Todo what is the correct upsampling here? (8,8) makes it work but we think it should be (2,2)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv8_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv8_2'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv8_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(2, (1, 1), activation='softmax', padding='same', name='pred'))
    # model.add(layers.Dense(2, activation='softmax', name='pred'))
    return model


def computeLoss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
    return loss

# def brg2lab(ims):
#     for image in range(len(ims)):
#         picture_rgb = cv2.resize(ims[image],(256, 256))
#         picture = rgb2lab(picture_rgb)
#         #print(ims[image].reshape(ims[image].shape[0],-1).shape)
#         #converted = cv.cvtColor(ims[image].reshape(ims[image].shape[0],-1),cv.COLOR_BGR2LAB)
#         #print(converted.shape)
#     return converted

def labTransform(img):
    Input = np.zeros((256, 256, 1))
    target = np.zeros((256, 256, 2))
    image = cv2.resize(img,(256, 256))
    lab_input = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2LAB)
    Input = img[:, :, :1]
    target = img[:, :, 1:]
    return lab_input, target 

def trainNetwork(epochs_val, bsize):
    #train = read_data.import_data('train', bsize)
    #val = read_data.import_data('validation', bsize)
    model = buildLayers((100,256,256,3))
    #lab = brg2lab(train)
    #print(model.summary())
    # Todo next step is to implement the probability distribution a,b to upsample the pictures so we can train with them
    SGD = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=["accuracy"])# https://keras.io/losses/, check at the bottom at the page, "categorical_crossentropy is another term for multi-class log loss."
    
    train_data_generator = ImageDataGenerator(rescale=1./255, preprocessing_function = labTransform)
    train_generator = train_data_generator.flow_from_directory(directory='dataset/data/train', target_size = (256,256), batch_size=bsize, class_mode=None)
    validate_data_generator = ImageDataGenerator(rescale=1./255, preprocessing_function = labTransform)
    validate_generator = validate_data_generator.flow_from_directory(directory='dataset/data/validate', target_size = (256,256), batch_size=bsize, class_mode=None)
    
#     for data_batch, labels_batch in train_generator:
#         print('data batch shape:', data_batch.shape)
#         print('labels batch shape:', labels_batch.shape)
#         break
                                                                     
    model.fit_generator(generator=train_generator, steps_per_epoch=100, epochs=epochs_val, validation_data=validate_generator, validation_steps=50)
                                                                     
    model.fit(train_x, train_y, epochs=epochs_val, batch_size=bsize, validation_data=(validation_x, validation_y))
    return model
