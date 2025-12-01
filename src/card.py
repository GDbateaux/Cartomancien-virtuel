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

        if self.label is not None:
            if self.confidence is not None:
                text = f'{self.label} ({self.confidence:.2f})'
            else:
                text = self.label

            x, y, _, _ = cv2.boundingRect(pts)
            text_y = max(0, y - 10)

            cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1)
