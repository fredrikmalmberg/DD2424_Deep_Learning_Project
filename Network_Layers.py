from keras import losses, models, layers, optimizers, regularizers


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
    model.add(layers.UpSampling2D((2, 2)))    # We upsample one step before the layer, and then use a (1,1) stride
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv8_1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv8_2'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", strides=(1, 1),
                            padding='same', kernel_regularizer=l2_reg, name='conv8_3'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(2, (1, 1), activation='softmax', padding='same', name='pred'))
    return model


def computeLoss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
    return loss


def trainNetwork(train_x, train_y, validation_x, validation_y, epochs_val, bsize):
    model = buildLayers(train_x.shape)
    # Todo next step is to implement the probability distribution a,b to upsample the pictures so we can train with them
    SGD = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=[
        "accuracy"])  # https://keras.io/losses/, check at the bottom at the page, "categorical_crossentropy is another term for multi-class log loss."
    model.fit(train_x, train_y, epochs=epochs_val, batch_size=bsize, validation_data=(validation_x, validation_y))
    return model
