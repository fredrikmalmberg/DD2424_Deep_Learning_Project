import read_data


def main():
    train = read_data.import_data('train')
    # print(train['x'][0])
    # read_data.print_picture(train[0][0])


if __name__ == '__main__':
    main()