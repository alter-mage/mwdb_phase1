import pickle
import sys
import os
import cv2
import moments_of_color
import elbp
import hog


def start(input_images_folder):
    images = {}
    for filename in os.listdir(input_images_folder):
        img = cv2.imread(os.path.join(input_images_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[filename.split('.')[0]]={
                'image': img,
                'color_moment': moments_of_color.get_cm_vector(img),
                'elbp': elbp.get_elbp_vector(img),
                'hog': hog.get_hog_vector(img)
            }
    with open('%s/dataset_meta.pickle' % input_images_folder, 'wb') as handle:
        pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    try:
        input_image_folder_path = sys.argv[1]
        if not os.path.isdir(input_image_folder_path):
            print('Folder does not exist')
        start(input_image_folder_path)
    except:
        print('Input folder missing from command line')
