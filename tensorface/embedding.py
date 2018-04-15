import os

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from tensorface.const import PRETREINED_MODEL_DIR

MODEL_FILE_NAME = '20180402-114759/20180402-114759.pb'

# to get Flask not complain
global tf
_tf = tf
global sess
sess = None

def load_model(pb_file, input_map=None):
    global _tf
    global sess
    if sess is None:
        sess = _tf.Session()
        print('Model filename: %s' % pb_file)
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = _tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _tf.import_graph_def(graph_def, input_map=input_map, name='')


load_model(os.path.join(PRETREINED_MODEL_DIR, MODEL_FILE_NAME))


# inception net requires this
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def embedding(face_np):
    global sess
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    x = prewhiten(face_np)
    feed_dict = {images_placeholder: [x], phase_train_placeholder: False}
    result = sess.run(embeddings, feed_dict=feed_dict)[0]
    return result


def input_shape():
    return _tf.get_default_graph().get_tensor_by_name("input:0").get_shape()


def embedding_size():
    return _tf.get_default_graph().get_tensor_by_name("embeddings:0").get_shape()[1]
