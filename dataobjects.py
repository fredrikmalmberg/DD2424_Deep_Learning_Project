from keras import regularizers


class settings:
    """
    Settings for creation of the model
    """

    def __init__(self, nr_colors_space):
        self.input_shape = (256, 256, 1)  # Dimensions of the input layer
        self.nr_colors_space = nr_colors_space  # Each color each picture can assume, set by the seen data set
        self.output_shape = (64, 64, nr_colors_space)  # Shape of the output
        self.regularizer = regularizers.l2(l=0.001)  # What regulizer to use in the layers, # Todo find the value of the regularization in the paper, default here is 0.01
        self.kernel_initializer = "he_normal"  # Initialization method of the layers

        # Training settings
        self.nr_epochs = 2
        self.training_steps_per_epoch = 10
        self.validation_steps_per_epoch = 10
        self.batch_size = 1  # Currently we can only run batch_size of 2 without getting out of memory error !!! This is with 8 GB VRAM !!!
        self.learning_rate = 0.001  # Learning rate of the training
        self.loss_function = "categorical_crossentropy"  # Which loss function to use

        # Test settings
        self.test_step_size = 50  # 700 Number iteration with batch size to traverse all data,  (data_samples//batch_size)
        self.from_checkpoint = False #true if loading from previous checkpoint

    def print_training_settings(self):
        print("Settings for the training:\nNumber of epochs: {nr_epochs}, Batch size: {batch_size}, "
              "training iterations per epoch: {training_steps_per_epoch}, this results in {nr_of_training_images} "
              "images being used for training. "
              "\nValidation iterations per epoch: {validation_steps_per_epoch}, this results in "
              "{nr_of_validation_images} images will used for training. "
              .format(nr_epochs=self.nr_epochs, batch_size=self.batch_size,
                      training_steps_per_epoch=self.training_steps_per_epoch,
                      nr_of_training_images=(self.batch_size * self.training_steps_per_epoch),
                      validation_steps_per_epoch=self.validation_steps_per_epoch,
                      nr_of_validation_images=(self.batch_size * self.validation_steps_per_epoch)))
