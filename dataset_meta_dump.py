import download_dataset
import moments_of_color
import elbp
import hog
import pickle

def start(pickle_path):
    images, targets=download_dataset.get_image_array()
    counter, image_id_map = 0, {}
    for image, target in zip(images, targets):
        current_id='image_%s_%s' % (target, counter % 10)
        image_id_map[current_id]={
            'image': image,
            'color_moment': moments_of_color.get_cm_vector(image),
            'elbp': elbp.get_elbp_vector(image),
            'hog': hog.get_hog_vector(image)
        }
        counter += 1
    with open(pickle_path, 'wb') as handle:
        pickle.dump(image_id_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    start('dataset_meta.pickle')