import warnings
import numpy as np
import matplotlib.pyplot as plt
import dataobjects
from Network_Layers import train_network, evaluate_model
import data_manager
from test_plot import plot_output, colorize_benchmark_images



warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    # unique_colors = np.load('dataset/data/color_space.npy')  # The list of unique color combinations
    # train = data_manager.import_data('train', 1)  # this is just for plotting
    settings = dataobjects.settings(313)
    data_manager.assert_data_is_setup(settings)  # Asserts that all necessary data files exists
    model = train_network(settings)
    # model_name = data_manager.save_model(model)
    # evaluate_model(model, settings)

    # loaded_model = data_manager.load_model("2020_05_10_16_02")
    colorize_benchmark_images(model)           # Benchmark the model by colorizing 3 defined pictures
    # plot_output(model, train['input'][0, :, :, :] , train['target'][0, :, :, :])
    # # evaluate_model(loaded_model, settings)


if __name__ == '__main__':
    main()
