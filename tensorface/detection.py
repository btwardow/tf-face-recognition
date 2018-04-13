import numpy as np
from mtcnn.mtcnn import MTCNN

detector = MTCNN()


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_faces(image, threshold=0.5):
    image_np = load_image_into_numpy_array(image)
    result = detector.detect_faces(image_np)
    return result
