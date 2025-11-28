import cv2
import numpy as np

from card import Card


class CardExtractor:
    def __init__(self, file_path, debug=False):
        self.file_path = file_path
        self.img = cv2.imread(file_path)
        self.debug = debug

        if self.img is None:
            raise ValueError(f'Could not read image from {file_path}')

    def _display(self, img, max_dim=1000):
        if not self.debug:
            return
        h, w = img.shape[:2]
        scale = min(max_dim / w, max_dim / h, 1.0)

        new_w = int(w * scale)
        new_h = int(h * scale)

        img = cv2.resize(img, (new_w, new_h))
        cv2.imshow('Displayed Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _preprocess(self):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self._display(img_gray)
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
        self._display(img_blur)
        img_thresh = cv2.adaptiveThreshold(
            img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 2)
        self._display(img_thresh)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        self._display(mask)
        return mask

    def _extract_boxes(self, mask):
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = mask.shape[:2]
        img_area = w * h

        min_area = img_area * 0.01
        max_area = img_area * 0.80

        card_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area or area < min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            epsilons = [0.01, 0.02, 0.03, 0.05]
            for e in epsilons:
                approx = cv2.approxPolyDP(cnt, e * peri, True)
                if len(approx) == 4:
                    card_boxes.append(approx)
                    break

        img_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).copy()
        img_contours = cv2.drawContours(img_contours, card_boxes, -1, (255, 0, 0), 3)
        self._display(img_contours)
        return card_boxes

    def get_cards(self) -> list[Card]:
        mask = self._preprocess()
        boxes = self._extract_boxes(mask)

        card_width = 300
        card_height = 600
        card_pt = np.float32([[0,0], [0,card_height], [card_width,card_height], [card_width,0]])

        cards = []
        for b in boxes:
            b = b.reshape(4, 2).astype(np.float32)

            length_1 = cv2.norm(b[0] - b[1])
            length_2 = cv2.norm(b[1] - b[2])

            if length_1 < length_2:
                b = np.concatenate([b[1:], [b[0]]])

            M = cv2.getPerspectiveTransform(b, card_pt)
            card = cv2.warpPerspective(self.img, M, (card_width,card_height))
            
            cards.append(Card(box=b, image=card))
        return cards
