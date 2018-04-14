import numpy as np
from mtcnn.mtcnn import MTCNN

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

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def get_faces(image, threshold=0.5, minsize=20, face_crop_size=160, face_crop_margin=32):
    img = load_image_into_numpy_array(image)
    # face detection parameters
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    faces = []

    bounding_boxes, _ = detect_face(img, minsize, pnet, rnet, onet,
                                                      threshold, factor)
    for bb in bounding_boxes:
        bb[2:4] -= bb[:2]
        faces.append(Face(*bb))

    return faces
