import cv2
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

def get_image_array():
    olivetti_faces_dataset = sklearn.datasets.fetch_olivetti_faces(
        data_home=None,
        shuffle=False,
        random_state=0,
        download_if_missing=True
    )

    images, targets = olivetti_faces_dataset['images'], olivetti_faces_dataset['target']
    return images, targets


# plt.imshow(
#     images[0],
#     cmap='gray'
# )
# # plt.savefig('DR.png')
# plt.show()

# cv2.imwrite('DR.png', edges_DR


def dump_dataset():
    olivetti_faces_dataset = sklearn.datasets.fetch_olivetti_faces(
        data_home=None,
        shuffle=False,
        random_state=0,
        download_if_missing=True
    )

    images, targets = olivetti_faces_dataset['images'], olivetti_faces_dataset['target']
    counter = 0
    for image, target in zip(images, targets):
        # cv2.imwrite('images/image_%s_%s.png' % (target, counter%10), image)
        plt.imsave(
            'images/image_%s_%s.png' % (target, counter % 10),
            image,
            cmap='gray'
        )
        counter += 1

    print(olivetti_faces_dataset)
    pass


if __name__=='__main__':
    dump_dataset()