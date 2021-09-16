import datetime
import math
import pickle
import sys
import os
import time

import dataset_meta_dump
import elbp
import hog
import moments_of_color
import similarity_metrics
import heapq
import matplotlib.pyplot as plt
import numpy as np

def start(input_image_folder, input_image_id, model, k):
    if not os.path.isdir(input_image_folder):
        print('Entered input image folder does not exist')
        return

    pickle_path='images/dataset_meta.pickle'
    while not os.path.isfile(pickle_path):
        print('pickle not found, run task 0 first')
        return

    with open(pickle_path, 'rb') as handle:
        image_map = pickle.load(handle)

    if input_image_id not in image_map:
        print('Invalid image ID')
        return
    current_image_descriptor=image_map[input_image_id]

    plt.imshow(
        current_image_descriptor['image'],
        cmap='gray'
    )
    plt.show()

    h=[]
    heapq.heapify(h)
    for filename in os.listdir(input_image_folder):
        if filename.split('.')[-1] != 'png':
            continue
        image = plt.imread(os.path.join(input_image_folder, filename))
        if model == 'color_moment':
            cm_similarity = similarity_metrics.l1_norm(current_image_descriptor['color_moment'],
                                                       moments_of_color.get_cm_vector(image))
            heapq.heappush(h, (-cm_similarity, image))
        elif model == 'elbp':
            elbp_similarity = similarity_metrics.correlation_coeff(current_image_descriptor['elbp'],
                                                                   elbp.get_elbp_vector(image))
            heapq.heappush(h, (-elbp_similarity, image))
        elif model == 'hog':
            hog_data = hog.get_hog_vector(image)
            hog_histogram_similarity = similarity_metrics.intersection(current_image_descriptor['hog']['hog_vector'],
                                                                    hog_data['hog_vector'])
            hog_image_similarity = similarity_metrics.cosine(current_image_descriptor['hog']['hog_image'],
                                                             hog_data['hog_image'])
            hog_scores = [hog_histogram_similarity, hog_image_similarity]
            hog_score = similarity_metrics.l2_norm1([float(i)/sum(hog_scores) for i in hog_scores])
            heapq.heappush(h, (-hog_score, image))
        if len(h) > k:
            heapq.heappop(h)

    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H_%M_%S')
    output_dir = os.path.join(input_image_folder, 'output_%s' % timestamp)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    fig, axs = plt.subplots(2, math.ceil(k/2))
    counter = 0
    while h:
        current_image=heapq.heappop(h)
        axs[counter%2, counter//2].imshow(current_image[1], cmap='gray')
        axs[counter%2, counter//2].set_title('Image score: %s' % str(-current_image[0]))
        counter += 1

        plt.imsave(
            '%s/%s.png' % (output_dir, 'score_%s' %  str(-current_image[0])),
            current_image[1],
            cmap='gray'
        )
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