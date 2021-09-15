import pickle
import sys
import os
import dataset_meta_dump
import similarity_metrics
import heapq
import matplotlib.pyplot as plt

def start(input_image_folder, input_image_id, k):
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
            cm_similarity=similarity_metrics.l1_norm(current_image_descriptor['color_moment'], image_map[image_key]['color_moment'])
            elbp_similarity=similarity_metrics.correlation_coeff(current_image_descriptor['color_moment'], image_map[image_key]['color_moment'])
            hog_histogram_similarity=similarity_metrics.linf_norm(current_image_descriptor['color_moment'], image_map[image_key]['color_moment'])
            hog_image_similarity=similarity_metrics.cosine(current_image_descriptor['color_moment'], image_map[image_key]['color_moment'])
            image_score=similarity_metrics.l2_norm1([cm_similarity, elbp_similarity, hog_histogram_similarity, hog_image_similarity])
            heapq.heappush(h, (-image_score, image_map[image_key]['image']))
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
        k=int(sys.argv[3])
    except:
        print('Invalid parameter entered.\nUsage: python3 task3.py [input image folder] [image ID] [k]')
    start(input_image_folder, input_image_id, k)