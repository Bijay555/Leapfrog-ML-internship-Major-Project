import cv2
import numpy as np

def image_mask(image):
    my_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    my_img= cv2.bilateralFilter(my_img,5, 55,60)
    my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(my_img, 240, 255, 1)
    return im