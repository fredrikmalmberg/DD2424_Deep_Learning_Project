import cv2
import os


def import_data(method):
    data = list()
    folders = os.listdir('dataset/data/{}/'.format(method))
    for subfolder in folders:
        entry = os.listdir('dataset/data/{}/{}'.format(method, subfolder))
        print(subfolder)
        for image in entry:
            picture = cv2.resize(cv2.imread('dataset/data/{}/{}/{}'.format(method, subfolder, image)), (256, 256))
            data.append(picture.reshape(-1, 1))

    return data


def print_picture(picture):
    pic = picture.reshape((256, 256, 3))
    cv2.imshow('image', pic)
    # not sure why but it need this 2 following rows to work
    cv2.waitKey(0)
    cv2.destroyAllWindows()
