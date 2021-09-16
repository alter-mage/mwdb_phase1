import os
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import pickle

import elbp
import hog
import moments_of_color


def dump_dataset():
    olivetti_faces_dataset = sklearn.datasets.fetch_olivetti_faces(
        data_home=None,
        shuffle=False,
        random_state=0,
        download_if_missing=True
    )

    images, targets = olivetti_faces_dataset['images'], olivetti_faces_dataset['target']
    counter = 0

    images_dir = os.path.join(os.getcwd(), 'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
    image_map = {}
    for image, target in zip(images, targets):
        image_id = 'image_%s_%s' % (target, counter % 10)
        plt.imsave(
            '%s/%s.png' % (images_dir, image_id),
            image,
            cmap='gray'
        )
        image_map[image_id] = {
            'image': image,
            'color_moment': moments_of_color.get_cm_vector(image),
            'elbp': elbp.get_elbp_vector(image),
            'hog': hog.get_hog_vector(image)
        }
        counter += 1
    pickle_path = '%s/dataset_meta.pickle' % images_dir
    with open(pickle_path, 'wb') as handle:
        pickle.dump(image_map, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=='__main__':
    dump_dataset()