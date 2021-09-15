import pickle
import sys
import os
import dataset_meta_dump
import similarity_metrics
import heapq
import matplotlib.pyplot as plt
import numpy as np

def start(input_image_folder, input_image_id, model, k):
    if not os.path.isdir(input_image_folder):
        print('Entered input image folder does not exist')
        return

    pickle_path=os.path.join(input_image_folder, 'dataset_meta.pickle')
    while not os.path.isfile(pickle_path):
        print('pickle not found, descriptor dump in progress')
        dataset_meta_dump.start(pickle_path)

    with open(pickle_path, 'rb') as handle:
        image_map = pickle.load(handle)

    if input_image_id not in image_map:
        print('Invalid image ID')
        return
    current_image_descriptor=image_map[input_image_id]



    h=[]
    heapq.heapify(h)
    for image_key in image_map:
        if image_key!=input_image_id:
            if model=='color_moment':
                cm_similarity=similarity_metrics.l1_norm(current_image_descriptor['color_moment'], image_map[image_key]['color_moment'])
                heapq.heappush(h, (-cm_similarity, image_map[image_key]['image']))
            elif model=='elbp':
                elbp_similarity=similarity_metrics.correlation_coeff(current_image_descriptor['elbp'], image_map[image_key]['elbp'])
                heapq.heappush(h, (-elbp_similarity, image_map[image_key]['image']))
            elif model=='hog':
                hog_histogram_similarity = similarity_metrics.linf_norm(current_image_descriptor['hog']['hog_vector'],
                                                                           image_map[image_key]['hog']['hog_vector'])
                hog_image_similarity = similarity_metrics.cosine(current_image_descriptor['hog']['hog_image'],
                                                                 image_map[image_key]['hog']['hog_image'])
                hog_score=similarity_metrics.l2_norm1([hog_histogram_similarity, hog_image_similarity])
                heapq.heappush(h, (-hog_score, image_map[image_key]['image']))
            if len(h)>k:
                heapq.heappop(h)

    while h:
        current_image=heapq.heappop(h)
        plt.imshow(current_image[1], cmap='gray')
        plt.show()

if __name__=='__main__':
    print(sys.argv)
    try:
        input_image_folder=sys.argv[1]
        input_image_id=sys.argv[2]
        model=sys.argv[3]
        if model not in ['color_moment', 'elbp', 'hog']:
            print('Invalid model entered, valid models are: color_moment, elbp, hog')
            sys.exit(0)
        k=int(sys.argv[4])
    except:
        print('Invalid parameter entered.\nUsage: python3 task3.py [input image folder] [image ID] [model name] [k]')
    start(input_image_folder, input_image_id, model, k)