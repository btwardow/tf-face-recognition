from collections import Counter
from typing import List, Tuple

import tensorflow as tf
import numpy as np

from tensorface.const import EMBEDDING_SIZE, UNKNOWN_CLASS


class KNN:
    def __init__(self, K=5, dist_threshold=14):
        '''
        Why such dist_threshold value?
        See notebook: notebooks/experiments_with_classification.ipynb
        :param K:
        :param dist_threshold:
        '''

        # current training data
        self.X_train = None
        self.y_train = None
        self.idx_to_lbl = None
        self.lbl_to_idx = None
        self.y_train_idx = None

        # main params
        self.dist_threshold_value = dist_threshold
        self.K = K

        # placeholders
        self.xtr = tf.placeholder(tf.float32, [None, EMBEDDING_SIZE], name='X_train')
        self.ytr = tf.placeholder(tf.float32, [None], name='y_train')
        self.xte = tf.placeholder(tf.float32, [EMBEDDING_SIZE], name='x_test')
        self.dist_threshold = tf.placeholder(tf.float32, shape=(), name="dist_threshold")

        ############ build model ############

        # model
        distance = tf.reduce_sum(tf.abs(tf.subtract(self.xtr, self.xte)), axis=1)
        values, indices = tf.nn.top_k(tf.negative(distance), k=self.K, sorted=False)
        nn_dist = tf.negative(values)
        self.valid_nn_num = tf.reduce_sum(tf.cast(nn_dist < self.dist_threshold, tf.float32))
        nn = []
        for i in range(self.K):
            nn.append(self.ytr[indices[i]])  # taking the result indexes

        # saving list in tensor variable
        nearest_neighbors = nn
        # this will return the unique neighbors the count will return the most common's index
        self.y, idx, self.count = tf.unique_with_counts(nearest_neighbors)
        self.pred = tf.slice(self.y, begin=[tf.argmax(self.count, 0)], size=tf.constant([1], dtype=tf.int64))[0]

    def predict(self, X) -> List[Tuple[str, float, List[str], List[float]]]:
        if self.X_train is None:
            # theres nothing we can do than just mark all faces as unknown...
            return [(UNKNOWN_CLASS, None, None, None) for _ in range(X.shape[0])]

        result = []
        if self.X_train is not None and self.X_train.shape[0] > 0:
            with tf.Session() as sess:
                for i in range(X.shape[0]):
                    _valid_nn_num, _pred, _lbl_idx, _counts = sess.run(
                        [self.valid_nn_num, self.pred, self.y, self.count],
                        feed_dict={
                            self.xtr: self.X_train,
                            self.ytr: self.y_train_idx,
                            self.xte: X[i, :],
                            self.dist_threshold: self.dist_threshold_value})

                    if _valid_nn_num == self.K:
                        s = _counts.sum()
                        c_lbl = []
                        c_prob = []
                        prob = None
                        for i, c in zip(_lbl_idx, _counts):
                            c_lbl.append(self.idx_to_lbl[i])
                            c_prob.append(float(c/s))
                            if _pred == i:
                                prob = float(c/s)

                        result.append((
                            self.idx_to_lbl[int(_pred)],
                            float(prob),
                            c_lbl,
                            c_prob
                        ))
                    else:
                        result.append((UNKNOWN_CLASS, None, None, None))

        return result

    def update_training(self, train_X, train_y):
        self.X_train = np.array(train_X)
        self.y_train = train_y
        self.idx_to_lbl = dict(enumerate(set(train_y)))
        self.lbl_to_idx = {v: k for k, v in self.idx_to_lbl.items()}
        self.y_train_idx = [self.lbl_to_idx[l] for l in self.y_train]


def init():
    global X, y, model
    X = []
    y = []
    model = KNN()


init()


def add(new_X, new_y):
    global X, y, model
    X.extend(new_X)
    y.extend(new_y)
    model.update_training(X, y)

def predict(X):
    global model
    return model.predict(X)

def training_data_info():
    global y
    return Counter(y)
