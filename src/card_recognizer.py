import cv2
import torch
import numpy as np
import torchvision.models as models

from typing import Union
from pathlib import Path
from numpy.linalg import norm
from torchvision import transforms


"""
Card recognition using image embeddings.

The embedding model and similarity approach are inspired by:
- Tayyib Ul Hassan Gondal, "Image Embeddings for Enhanced Image Search"
  https://medium.com/thedeephub/image-embeddings-for-enhanced-image-search-f35608752d42
- PyTorch Hub ResNet example
  https://pytorch.org/hub/pytorch_vision_resnet/

The SIFT part is inspired by:
- https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
- https://stackoverflow.com/questions/50217364/sift-comparison-calculate-similarity-score-python
"""
class CardRecognizer:
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

        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.bf = cv2.BFMatcher()
        self.img_size = (300, 600)

        self.ref_embeddings = self._load_embeddings()
        self.ref_sift = self._load_sift_descriptors()

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

    def _load_sift_descriptors(self):
        image_paths = sorted(
            list(self.ref_dir.glob('*.jpg')) +
            list(self.ref_dir.glob('*.png'))
        )

        descriptors = {}
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            kp, desc = self._img_to_sift(img)
            descriptors[img_path.stem] = (kp, desc)
        return descriptors
    
    def _img_to_sift(self, img):
        img = cv2.resize(img, self.img_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray,None)
        return keypoints, descriptors

    def _sift_score(self, kp, desc, ref_kp, ref_desc, ratio=0.75):
        if desc is None or ref_desc is None:
            return 0
        matches = self.bf.knnMatch(desc, ref_desc, k=2)

        good = []
        for m,n in matches:
            if m.distance < ratio * n.distance:
                good.append([m])
        return len(good) * 100 / len(kp) if len(kp) > 0 else 0
    
    def recognize(self, img, top_k: int = 1, min_score: float = None):
        img_flip = cv2.rotate(img, cv2.ROTATE_180)
        emb1 = self._img_to_embedding(img)
        emb2 = self._img_to_embedding(img_flip)
        
        embed_scores = []

        for label, embed_ref in self.ref_embeddings.items():
            score1 = float(np.dot(emb1, embed_ref))
            score2 = float(np.dot(emb2, embed_ref))
            embed_scores.append((label, max(score1, score2)))
        embed_scores.sort(key=lambda x: x[1], reverse=True)

        if min_score is not None:
            embed_scores = [s for s in embed_scores if s[1] >= min_score]
        
        if not embed_scores:
            return []
        
        N = min(max(top_k * 3, 6), len(embed_scores))
        candidates = embed_scores[:N]
        kp_image, desc_image = self._img_to_sift(img)

        hybrid_scores = []
        for label, _ in candidates:
            kp_ref, desc_ref = self.ref_sift[label]
            hybrid_scores.append((label, self._sift_score(kp_image, desc_image, kp_ref, desc_ref)))
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)

        return hybrid_scores[:top_k]
