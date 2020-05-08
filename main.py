import warnings

import dataobjects
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
from Network_Layers import train_network


def main():
    # TODO add check here to see if unique list file exists else create it. Also other checks like if dat aexits and so on

    unique_colors = np.load('dataset/data/color_space.npy')

    print("Starting training of network")
    settings = dataobjects.settings(313)  # TODO this output value should be set bu the unique list
    model = train_network(settings, unique_colors)
    print("Training done")


if __name__ == '__main__':
    main()
