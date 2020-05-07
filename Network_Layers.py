import tensorflow as tf
from keras import losses, models, layers, optimizers, regularizers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # This line allows the network to use the GPU VRAM uncapped. !!! NEED THIS LINE FOR NETWORK TO RUN !!!
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    except RuntimeError as e:
        print(e)


def build_model(dimensions):
    """
    Build a network according to Image Colorization paper: https://arxiv.org/abs/1603.08511
    We use He initialization because it tends to give better results when using ReLu activation function.
    :param dimensions: Input dimensions of the data
    :return: Returns a Keras objective model
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
    model.add(layers.UpSampling2D(
        (8, 8)))  # Todo what is the correct upsampling here? (8,8) makes it work but we think it should be (2,2)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv8_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv8_2'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv8_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(2, (1, 1), activation='softmax', padding='same', name='pred'))
    return model


def compute_loss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
    return loss


def train_network(train_x, train_y, validation_x, validation_y, epochs_val, bsize):
    model = build_model(train_x.shape)
    print(model.summary())
    # Todo next step is to implement the probability distribution a,b to upsample the pictures so we can train with them
    SGD = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=[
        "accuracy"])  # https://keras.io/losses/, check at the bottom at the page, "categorical_crossentropy is another term for multi-class log loss."
    model.fit(train_x, train_y, epochs=epochs_val, batch_size=bsize, validation_data=(validation_x, validation_y))  # Todo we gonna have to change to model.fit_generatot(to stream the entire data set while training)
    return model
