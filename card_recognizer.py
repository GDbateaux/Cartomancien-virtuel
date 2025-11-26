import cv2
import torch
import numpy as np
import torchvision.models as models

from typing import Union
from pathlib import Path
from numpy.linalg import norm
from torchvision import transforms


class CardRecognizer():
    def __init__(self, ref_dir: Union[str, Path]):
        self.ref_dir: Path = Path(ref_dir)

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.ref_embeddings = self._load_embeddings()

    def _load_embeddings(self):
        image_paths = sorted(
            list(self.ref_dir.glob('*.jpg')) +
            list(self.ref_dir.glob('*.png'))
        )

        embeddings = {}
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            emb = self._img_to_embedding(img)
            embeddings[img_path.stem] = emb
        return embeddings

    # OpenCV image
    def _img_to_embedding(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(img_rgb)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = self.model(input_batch)
        
        emb = embedding.squeeze().numpy()
        n = norm(emb)
        if n > 0:
            emb = emb / n
        return emb
    
    def recognize(self, img, top_k: int = 1):
        img_flip = cv2.rotate(img, cv2.ROTATE_180)
        emb1 = self._img_to_embedding(img)
        emb2 = self._img_to_embedding(img_flip)
        
        scores = []

        for k, v in self.ref_embeddings.items():
            score1 = float(np.dot(emb1, v))
            score2 = float(np.dot(emb2, v))
            scores.append((k, max(score1, score2)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
