import warnings

import dataobjects
from Network_Layers import train_network, evaluate_model
# from data_manager import assert_data_is_setup, save_model, load_model
import data_manager

warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    data_manager.assert_data_is_setup()  # Asserts that all necessary data files exists
    settings = dataobjects.settings(313)
    # model = train_network(settings)
    # evaluate_model(model, settings)

    # model_name = data_manager.save_model(model)
    loaded_model = data_manager.load_model("model_15_epochs_on_complete_data_set")
    evaluate_model(loaded_model, settings)


if __name__ == '__main__':
    main()
