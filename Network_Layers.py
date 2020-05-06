import keras
from keras import losses, models, layers, optimizers


def buildLayers(dimensions):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=dimensions[1:]))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='softmax'))
    return model


def computeLoss(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
    return loss


def trainNetwork(train_x, train_y, validation_x, validation_y, epochs_val, bsize):
    model = buildLayers(train_x.shape)
    SGD = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=epochs_val, batch_size=bsize, validation_data=(validation_x, validation_y))
    return model
