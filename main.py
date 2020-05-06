import read_data
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
from Network_Layers import buildLayers, computeLoss, trainNetwork


def main():
    ''' RUN THE NEXT 2 ROWS TO GET THE DATA IN LAB SHAPE LOCALLY (NEED TO RUN ONLY ONCE)'''
    # read_data.save_lab_figures('train')
    # read_data.save_lab_figures('validation')
    ''' IMPORTING THE LAB FILES'''
    train = read_data.import_data('train', 100)
    validation = read_data.import_data('validation', 100)
    # read_data.print_picture(train['input'][0])
    tensor_data = tf.data.Dataset.from_tensor_slices((train['input'], train['target']))
    model = trainNetwork(train['input'], train['target'], validation['input'], validation['target'], epochs_val=2,bsize=100)
    # print(tensor_data)


if __name__ == '__main__':
    main()
