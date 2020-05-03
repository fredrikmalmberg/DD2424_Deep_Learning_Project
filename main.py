import read_data
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf


def main():
    # read_data.save_lab_figures('train')
    train = read_data.import_data('train')
    # validation = read_data.import_data('validation')
    # read_data.print_picture(train['x'][0])
    tensor_data = tf.data.Dataset.from_tensor_slices((train['x'], train['y']))
    print(tensor_data)


if __name__ == '__main__':
    main()
