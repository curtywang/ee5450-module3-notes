import numpy as np
import matplotlib.pyplot as plt
import cv2
# from skimage import io


def main():
    cat_image: np.ndarray = cv2.imread('cat.png')  # cv2 will always read images in as BGR

    cat_image = cv2.cvtColor(cat_image, cv2.COLOR_BGR2RGB)

    # TODO: How do we get the top-left quadrant only of cat_image?
    cat_dims = cat_image.shape  # cat_dims = (720, 1280, 3)
    top_left_cat_image = np.copy(cat_image[:(cat_dims[0] // 2), :(cat_dims[1] // 2), :])

    cat_image = cat_image.astype(np.int32)
    cat_image = cat_image - 20
    cat_image = np.clip(cat_image, a_min=0, a_max=255).astype(np.uint8)
    # cat_image[:(cat_dims[0] // 2), :(cat_dims[1] // 2), :] = 0  # black out the top left

    # io.imshow(cat_image)
    # io.imshow(top_left_cat_image)
    # io.show()

    return


if __name__ == '__main__':
    main()
