# MWDB Phase 1
## alter_mage's image retrieval system

This is the phase 1 of my MWDB project. This is an image retrieval system which looks at three models, namely, Color Moment, Extended Local Binary Patterns and Histogram of Oriented Gradients to retrieve images based on similarity metrics.

## Features

- Modularized scripts for different tasks
- Provision of a standalone model retrieval or an ensemble retrieval
- Tested on Olivetti Faces Dataset
- No need for database configuration, just a simple plug and play!

The scripts are extremely and intuitive on their own.

## Assumptions
 - We expect all the images in the input image folder to be 64x64 grayscale images inline with the Olivetti Faces dataset
 - All the images need to be in png format

## Requirements
 - Preferably a Unix system
 - Python 3.6+
 - That's really it!

## Setup instructions
After cloning
```sh
pip3 install -r requirements.txt
```

## Plug and play
Download and dump the dataset using
```sh
python3 task0.py
```

Display the feature descriptors of any image
```sh
python3 task1.py <image_id>
```

Store the feature descriptors of a particular folder containing images
```sh
python3 task2.py <absolute path of the folder containing images>
```

Retrieve k similar images from the query image using a particular model from the provided input folder containing images
```sh
python3 task3.py <absolute path of the folder containing images> <query image ID> <model name> <k: number of similar images>
```

Retrieve k similar images from the query image using all three models from the provided input folder containing images
```sh
python3 task4.py <absolute path of the folder containing images> <query image ID> <k: number of similar images>
```

For image retrieval executions,  the retrieved scripts will be dumped in the output_<timestamp> folder in the input images folder provided at the time of execution.