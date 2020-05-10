import warnings

import dataobjects
from Network_Layers import train_network, evaluate_model
import data_manager
import read_data
from test_plot import plot_output

warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    train = read_data.import_data('train', 10)  # this is just for plotting
    settings = dataobjects.settings(313)
    data_manager.assert_data_is_setup(settings)  # Asserts that all necessary data files exists
    model = train_network(settings)
    evaluate_model(model, settings)

    model_name = data_manager.save_model(model)
    # loaded_model = data_manager.load_model("model_15_epochs_on_complete_data_set")
    # evaluate_model(loaded_model, settings)
    # plot_output(loaded_model, train['input'][0, :, :, :], train['target'][0, :, :, :])


if __name__ == '__main__':
    main()
