import warnings
import numpy as np
import dataobjects
from Network_Layers import train_network
import data_manager
warnings.filterwarnings('ignore', category=FutureWarning)
import class_rebalance
import plotting


def main():
    priors = np.load('trained_models/prior_dog.npy')
    w = class_rebalance.get_re_weights(priors, lamb=0.5)

    settings = dataobjects.settings(priors.shape[0])
    data_manager.assert_data_is_setup(settings)  # Asserts that all necessary data files exists
    model = train_network(settings, w)
    # model_name = data_manager.save_model(model)
    # evaluate_model(model, settings)

    # loaded_model = data_manager.load_model("trained_models/2020_05_11_23_10")
    # colorize_benchmark_images(loaded_model)           # Benchmark the model by colorizing 3 defined pictures
    # # evaluate_model(loaded_model, settings)


if __name__ == '__main__':
    main()
