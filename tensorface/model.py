

class Face:
    # face bounding boxes
    def __init__(self, x, y, w, h, confidence):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.confidence = confidence

    def __repr__(self):
        return repr(self.__dict__)