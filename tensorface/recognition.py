from typing import List

import numpy as np
from PIL import Image

from tensorface import classifier
from tensorface.const import FACE_PIC_SIZE, EMBEDDING_SIZE
from tensorface.detection import img_to_np
from tensorface.embedding import embedding
from tensorface.model import Face


def recognize(faces) -> List[Face]:
    X = np.zeros((len(faces), EMBEDDING_SIZE), np.float32)
    for i, f in enumerate(faces):
        img = f.img.resize((FACE_PIC_SIZE, FACE_PIC_SIZE), Image.BICUBIC) if f.img.size != (FACE_PIC_SIZE,
                                                                                            FACE_PIC_SIZE) else f.img

        X[i, :] = embedding(img_to_np(img))

    result = classifier.predict(X)
    for f, r in zip(faces, result):
        n, prob, c_list, c_prob = r
        f.name = n
        f.predict_proba = prob
        f.predict_candidates = c_list
        f.predict_candidates_proba = c_prob

    return faces


def learn_from_examples(name, image_sprite, num, size):

    print("Adding new training data for: ", name, "...")

    # update classifier
    faces = []
    for i in range(int(num)):
        faces.append(image_sprite.crop((
            size * i,
            0,
            size * (i + 1),
            size
        )))

    # do embedding for all faces
    X = np.zeros((num, EMBEDDING_SIZE), np.float32)
    for i, f in enumerate(faces):
        X[i, :] = embedding(img_to_np(f))

    # all example cames from single person
    y = [name] * num

    # do the actual update
    classifier.add(X, y)

    return classifier.training_data_info()
