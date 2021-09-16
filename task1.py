import pickle
import argparse
import sys

import matplotlib.pyplot as plt
import datetime
import time
import json

def start(image_id, model):
    with open('images/dataset_meta.pickle', 'rb') as handle:
        image_map = pickle.load(handle)
    if image_id not in image_map:
        print('invalid image ID, please check')
        return
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H_%M_%S')
    dump_file = '%s_%s_%s.json' % (image_id, model, timestamp)
    if model=='color_moment':
        with open(dump_file, 'w+') as f:
            json.dump({
                'data': image_map[image_id]['color_moment'].tolist()
            }, f)
    elif model=='elbp':
        with open(dump_file, 'w+') as f:
            json.dump({
                'data': image_map[image_id]['elbp'].tolist()
            }, f)
    elif model=='hog':
        hog_data = image_map[image_id]['hog']
        plt.imshow(hog_data['hog_image'], cmap='gray')
        plt.show()
        with open(dump_file, 'w+') as f:
            json.dump({
                'data': hog_data['hog_vector'].tolist()
            }, f)
    else:
        print('model not available')


if __name__=='__main__':
    image_id = sys.argv[1]
    model = sys.argv[2]
    start(image_id, model)