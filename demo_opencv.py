import numpy as np
import matplotlib.pyplot as plt
import cv2  # C/C++/python image processing library
# from skimage import io  # python-only image processing library


def main():
    cat_image: np.ndarray = cv2.imread('cat.png')  # cv2 will always read images in as BGR

    cat_image = cv2.cvtColor(cat_image, cv2.COLOR_BGR2RGB)  # converting from BGR to RGB order

    # get only the top left quadrant
    cat_dims = cat_image.shape  # cat_dims = (720, 1280, 3)
    top_left_cat_image = np.copy(cat_image[:(cat_dims[0] // 2), :(cat_dims[1] // 2), :])

    # brute-force darkening/brightening
    cat_image = cat_image.astype(np.int32)
    cat_image = cat_image - 20
    cat_image = np.clip(cat_image, a_min=0, a_max=255).astype(np.uint8)

    # black out the top left
    # cat_image[:(cat_dims[0] // 2), :(cat_dims[1] // 2), :] = 0

    # blur image
    # blurred_image = cv2.GaussianBlur(cat_image, (5, 5), 10.0)
    # cat_stack = np.hstack((cat_image, blurred_image))

    # thresholding for infrared thermometer
    gray_cat_image = cv2.cvtColor(cat_image, cv2.COLOR_RGB2GRAY)
    # thresholded_cat_image = cv2.adaptiveThreshold(gray_cat_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                               cv2.THRESH_BINARY_INV, 11, 2)
    retval, thresholded_cat_image = cv2.threshold(gray_cat_image, 228, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # mask with thresholded image
    masked_cat_image = cv2.bitwise_and(cat_image, cat_image, mask=thresholded_cat_image)

    plt.figure(figsize=(10, 4))
    plt.imshow(masked_cat_image, cmap='gray')
    # plt.imshow(cat_stack)
    # plt.imshow(top_left_cat_image)
    plt.show()

    return


if __name__ == '__main__':
    main()
