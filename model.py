from keras import models, layers, optimizers
from keras import backend as K
import tensorflow as tf


def create_model(settings, class_weights, training=True):
    """
    Creates a model and compiles it with parameters from the settings file
    :param model_name: Name of the model to load
    :param training: If we are training the network or using it for prediction
    :param class_weights:
    :param settings: Settings for the network
    :return: A compiled model ready to be trained
    """

    if settings.from_checkpoint:
        model = models.load_model(settings.checkpoint_filepath)
        print('successfully loaded checkpoint: {model_name}'.format(model_name=settings.checkpoint_filepath))
    else:
        regulizer = settings.regularizer
        initializer = settings.kernel_initializer

        class BetterSoftmax(layers.Layer):
            def __init__(self):
                super(BetterSoftmax, self).__init__()

            def call(self, inputs):
                ret = K.softmax(inputs, axis=-1)
                return ret

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
                                padding='same', kernel_regularizer=regulizer, name='conv4_1'))
        model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                                padding='same', kernel_regularizer=regulizer, name='conv4_2'))
        model.add(layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                                padding='same', kernel_regularizer=regulizer, name='conv4_3'))
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
                                padding='same', kernel_regularizer=regulizer, name='conv7_1'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                                padding='same', kernel_regularizer=regulizer, name='conv7_2'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                                padding='same', kernel_regularizer=regulizer, name='conv7_3'))
        model.add(layers.BatchNormalization())
        model.add(layers.UpSampling2D(size=(2, 2), name='upsample_1'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                                padding='same', kernel_regularizer=regulizer, name='conv8_1'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                                padding='same', kernel_regularizer=regulizer, name='conv8_2'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer, strides=(1, 1),
                                padding='same', kernel_regularizer=regulizer, name='conv8_3'))
        model.add(layers.Conv2D(settings.nr_colors_space, (1, 1), activation=None, kernel_initializer='glorot_normal',
                                strides=(1, 1), padding='same', dilation_rate=1, name='pred'))
        model.add(BetterSoftmax())

    if not training:  # this is run when we are predicting
        model.add(layers.UpSampling2D((4, 4), interpolation='bilinear', name='upsample_2_predict'))

    def loss_function(y_true, y_pred):
        if settings.use_rebalancing:
            c_w = tf.convert_to_tensor(class_weights)
            q_star = K.argmax(y_true, axis=-1)
            w_q = tf.gather(c_w, q_star)
            y_pred_log = K.log(y_pred + K.epsilon())
            ret = -K.sum(tf.multiply(w_q, K.sum(tf.multiply(y_true, y_pred_log), axis=-1)))
        else:
            y_pred_log = K.log(y_pred + K.epsilon())
            ret = -K.sum(tf.multiply(y_true, y_pred_log))
        return ret

    adam = optimizers.Adam(learning_rate=settings.learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-07)
    model.compile(loss=loss_function, optimizer=adam, metrics=["accuracy"])
    return model
