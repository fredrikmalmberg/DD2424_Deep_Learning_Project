import warnings
import numpy as np
import dataobjects
from Network_Layers import evaluate_model, get_callback_functions, train_pretrained_model
import data_manager

warnings.filterwarnings('ignore', category=FutureWarning)
import class_rebalance
import plotting


def main():
    priors = np.load('trained_models/prior_dog.npy')
    w = class_rebalance.get_re_weights(priors, lamb=0.5)

    settings = dataobjects.settings(priors.shape[0])
    data_manager.assert_data_is_setup(settings)  # Asserts that all necessary data files exists
    # model = train_network(settings, w)
    # model_name = data_manager.save_model(model)

    loaded_model = data_manager.load_model("checkpoints/download_checkpoint")
    plotting.colorize_images_in_folder(settings, loaded_model, w, "dataset/colorize_images/")

    # How to continue training
    # model = load_model("checkpoints/best_weights_saved", settings, w)
    # model = train_pretrained_model(model, settings, w)


if __name__ == '__main__':
    main()
