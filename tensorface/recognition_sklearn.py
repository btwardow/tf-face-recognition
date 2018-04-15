from typing import List

from PIL import Image
from sklearn.externals import joblib

from tensorface.const import FACE_PIC_SIZE
from tensorface.detection import img_to_np
from tensorface.embedding import embedding
from tensorface.model import Face

UNKNOW_CLASS = "unknown"

model = joblib.load('/Users/b.twardowski/Development/tf-face-recognition/notebooks/knn_test.model')


def recognize(faces) -> List[Face]:
    for f in faces:
        img = f.img.resize((FACE_PIC_SIZE, FACE_PIC_SIZE), Image.BICUBIC) if f.img.size != (FACE_PIC_SIZE,
                                                                                            FACE_PIC_SIZE) else f.img

        e = embedding(img_to_np(img))
        x = e.reshape([1, -1])
        n = model.predict(x)[0]
        f.name = n
        f.predict_proba = model.predict_proba(x).tolist()
    return faces
