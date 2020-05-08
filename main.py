import warnings

import dataobjects
from Network_Layers import train_network, evaluate_model
from data_manager import assert_data_is_setup, save_model, load_model

warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    assert_data_is_setup()  # Asserts that all necessary data files exists
    settings = dataobjects.settings(313)
    model = train_network(settings)
    evaluate_model(model, settings)

    # Todo fix this code, we cannot save or load models at the moment
    # save_model(model, "test")
    # loaded_model = load_model("test")
    # evaluate_model(loaded_model, settings)

if __name__ == '__main__':
    main()
