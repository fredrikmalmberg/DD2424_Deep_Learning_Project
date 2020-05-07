import warnings

import read_data

warnings.filterwarnings('ignore', category=FutureWarning)
from Network_Layers import train_network


def main():
    ''' RUN THE NEXT 2 ROWS TO GET THE DATA IN LAB SHAPE LOCALLY (NEED TO RUN ONLY ONCE)'''
    # read_data.save_lab_figures('train')
    # read_data.save_lab_figures('validation')
    ''' IMPORTING THE LAB FILES'''
    train = read_data.import_data('train', 100)
    validation = read_data.import_data('validation', 100)
    # read_data.print_picture(train['input'][0])
    # tensor_data = tf.data.Dataset.from_tensor_slices((train['input'], train['target']))

    # Currently we can only run batch_size of 2 without getting out of memory error !!! This is with 8 GB VRAM !!!
    print("Starting training of network")
    model = train_network(train['input'], train['target'], validation['input'], validation['target'], epochs_val=5, bsize=2)
    print("Training done")


if __name__ == '__main__':
    main()
