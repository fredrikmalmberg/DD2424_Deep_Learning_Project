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
    model_name = data_manager.save_model(model)


    # loaded_model = data_manager.load_model("checkpoints/mine_night_run")
    # plotting.colorize_images_in_folder(settings, loaded_model, w, "dataset/dogs/colorize_images/")
    # evaluate_model(model, settings) # See the test accuracy

if __name__ == '__main__':
    main()
