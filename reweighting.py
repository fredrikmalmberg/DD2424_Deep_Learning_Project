import numpy as np


def re_weight(target, lamda):
    '''
    function for re balacing the probabilities of the colors for each pixel
    :param target: array of 1 x Height x Width x 313
    :return: y: array of 1 x Height x Width x 313
    '''
    q = target.shape[3]
    reweighted = (1 - lamda) * target + lamda / q

    return np.power(reweighted, -1)


# target = np.random.normal(0, 0.1, (1, 256, 256, 313))
#
# y = re_weight(target, 0.5)
