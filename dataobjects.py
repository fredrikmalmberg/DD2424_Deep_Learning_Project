from keras import regularizers


class settings:
    """
    Settings for creation of the model
    """

    def __init__(self, nr_output_classes):
        self.input_shape = (256, 256, 3)            # Dimensions of the input layer
        self.nr_output_classes = nr_output_classes  # Dimension of the output layer
        self.regularizer = regularizers.l2()        # What regulizer to use in the layers, # Todo find the value of the regularization in the paper, default here is 0.01
        self.kernel_initializer = "he_normal"       # Initialization method of the layers

        # Training settings
        self.nr_epochs = 2
        self.training_steps_per_epoch = 2500
        self.validation_steps_per_epoch = 250
        self.batch_size = 2                              # Currently we can only run batch_size of 2 without getting out of memory error !!! This is with 8 GB VRAM !!!
        self.learning_rate = 0.001                       # Learning rate of the training
        self.loss_function = "categorical_crossentropy"  # Which loss function to use
