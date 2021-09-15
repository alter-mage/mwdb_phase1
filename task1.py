import pickle
import argparse
import matplotlib.pyplot as plt

def start(image_id, model):
    with open('dataset_meta.pickle', 'rb') as handle:
        image_map = pickle.load(handle)
    if model=='color_moment':
        plt.subplot(1, 2, 1)
        plt.imshow(image_map[image_id]['image'], cmap='gray')
        plt.show()
    elif model=='elbp':
        pass
    elif model=='hog':
        pass
    else:
        print('model not available')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Task 1')
    parser.add_argument(
        '-i',
        dest='image_id',
        help='image id',
        required=True
    )
    parser.add_argument(
        '-m',
        dest='model',
        help='model name as color_moment, elbp and hog',
        required=True
    )
    args=parser.parse_args()
    start(args.image_id, args.model)