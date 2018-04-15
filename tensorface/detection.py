import numpy as np
import time

from tensorface.const import PRETREINED_MODEL_DIR
from tensorface.mtcnn import detect_face, create_mtcnn
import tensorflow as tf

from tensorface.model import Face


def _setup_mtcnn():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            return create_mtcnn(sess, PRETREINED_MODEL_DIR)


pnet, rnet, onet = _setup_mtcnn()


def img_to_np(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_faces(image, threshold=0.5, minsize=20):
    img = img_to_np(image)
    # face detection parameters
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    faces = []

    bounding_boxes, _ = detect_face(img, minsize, pnet, rnet, onet,
                                                      threshold, factor)
    for bb in bounding_boxes:
        img = image.crop(bb[:4])
        bb[2:4] -= bb[:2]
        faces.append(Face(*bb, img))

    return faces
