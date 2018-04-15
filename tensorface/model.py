

class Face:
    # face bounding boxes
    def __init__(self, x, y, w, h, confidence, img):
        self.img = img
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence

        self.predict_proba = None
        self.predict_candidates = None
        self.predict_candidates_proba = None

    def data(self):
        return { k:v for k, v in self.__dict__.items() if k != 'img'}

