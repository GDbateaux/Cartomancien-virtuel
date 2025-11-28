import cv2
import numpy as np


class Card():
    def __init__(self, box, image, label=None, confidence=None):
        self.box = box
        self.image = image
        self.label = label
        self.confidence = confidence
    
    def draw_on(self, img, color=(0, 255, 0), thickness=2):
        pts = self.box.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
