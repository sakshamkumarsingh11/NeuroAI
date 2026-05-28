
import cv2
import numpy as np

def window_ct(img):
    img = img.astype(np.float32)
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip(img, p2, p98)
    img = (img - p2) / (p98 - p2 + 1e-6)
    return img

def preprocess_ct(file_bytes):
    file_bytes = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    original = cv2.resize(img, (224, 224))
    img = window_ct(original)

    img = np.stack([img, img, img], axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)

    return img, original
