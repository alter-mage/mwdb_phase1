import numpy as np
import pandas as pd
import skimage.feature


def get_elbp_df(images):
    elbp_df=[]
    for image in images:
        rotation_invariant_lbp = skimage.feature.local_binary_pattern(
            image,
            P=8,
            R=2,
            method='ror'
        )
        lbp_ror_histogram, _ = np.histogram(
            rotation_invariant_lbp,
            bins=10
        )
        lbp_ror_variance = np.var(rotation_invariant_lbp)

        lbp_ror_histogram=lbp_ror_histogram/lbp_ror_variance
        norm = np.linalg.norm(lbp_ror_histogram)
        normal_lbp_ror_histogram = lbp_ror_histogram/norm

        elbp_df.append(normal_lbp_ror_histogram)

    print('here')
    return pd.DataFrame(
        np.array(elbp_df, dtype=np.float32)
    )

def get_elbp_vector(image):
    rotation_invariant_lbp = skimage.feature.local_binary_pattern(
        image,
        P=8,
        R=2,
        method='ror'
    )
    return np.array(rotation_invariant_lbp, dtype=np.float32)

    # lbp_ror_histogram, _ = np.histogram(
    #     rotation_invariant_lbp,
    #     bins=10
    # )
    # norm = np.linalg.norm(lbp_ror_histogram)
    # normal_lbp_ror_histogram = lbp_ror_histogram / norm
    # return np.array(normal_lbp_ror_histogram, dtype=np.float32).ravel()