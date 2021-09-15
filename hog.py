import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.feature

def get_hog_df(images):
    # creating hog features
    hog_df=[]
    for image in images:
        hog_vector = skimage.feature.hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            multichannel=False
        )
        hog_df.append(hog_vector)
    return pd.DataFrame(np.array(
        hog_df,
        dtype=np.float32
    ))

def get_hog_vector(image):
    hog_vector, hog_image = skimage.feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        multichannel=False
    )
    return {
        'hog_vector': np.array(hog_vector, dtype=np.float32).ravel(),
        'hog_image': np.array(hog_image, dtype=np.float32)
    }
