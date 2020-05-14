from keras import regularizers


class settings:
    """
    Settings for creation of the model
    """

    def __init__(self, nr_colors_space):
        # Network settings
        self.input_shape = (224, 224, 1)  # Dimensions of the input layer
        self.input_layer_shape = (None, None, 1)
        self.nr_colors_space = nr_colors_space  # Each color each picture can assume, set by the seen data set
        self.output_shape = (56, 56, nr_colors_space)  # Shape of the output
        self.regularizer = regularizers.l2(l=1e-3)  # What regulizer to use in the layers
        self.kernel_initializer = "he_normal"  # Initialization method of the layers

        # Training settings
        self.plot_during_training = False
        self.plot_random_imgs_from_generator = False
        self.plot_every_n_batch = 5
        self.nr_epochs = 60
        self.training_steps_per_epoch = 1764     # Nr dogs training images after split = 12348/7 = 1764
        self.validation_steps_per_epoch = 882     # Nr dogs validation images after split = 6174/7 = 882
        self.batch_size = 7  # Currently we can only run batch_size of 2 without getting out of memory error !!! This is with 8 GB VRAM !!!
        self.loss_function = "categorical_crossentropy"  # Which loss function to use
        self.learning_rate = 3e-5  # Learning rate of the training
        self.min_learning_rate = 3e-6  # The minimum the learning rate can reduce to
        self.learning_rate_reduction = 0.5  # How much the learning rate will be reduced, new_lr = lr * factor
        self.patience = 10    # How many epochs it will wait for an improvement before triggering change

        # Test settings
        self.from_checkpoint = False  # True if loading from previous checkpoint
        self.checkpoint_filepath = "checkpoints/2020_05_11_22_51"   # The path and name for the model
        self.test_step_size = 10      # Number iteration with batch size to traverse all data,  (data_samples//batch_size)

        # self.data_directory = "../toy_data/"
        self.data_directory = "dataset/dogs/"
        # self.data_directory = "../data/ILSVRC/Data/CLS-LOC/"

    def print_training_settings(self):
        print("Settings for the training:\nNumber of epochs: {nr_epochs}, Batch size: {batch_size}, "
              "training iterations per epoch: {training_steps_per_epoch}, this results in {nr_of_training_images} "
              "images being used for training. "
              "\nValidation iterations per epoch: {validation_steps_per_epoch}, this results in "
              "{nr_of_validation_images} images will used for validation. "
              .format(nr_epochs=self.nr_epochs, batch_size=self.batch_size,
                      training_steps_per_epoch=self.training_steps_per_epoch,
                      nr_of_training_images=(self.batch_size * self.training_steps_per_epoch),
                      validation_steps_per_epoch=self.validation_steps_per_epoch,
                      nr_of_validation_images=(self.batch_size * self.validation_steps_per_epoch)))
