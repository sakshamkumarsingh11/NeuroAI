
import torch
import numpy as np
from model import TBIModel
from preprocess import preprocess_ct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TBIModel().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

classes = [
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural"
]

thresholds = np.array([0.45, 0.5, 0.5, 0.45, 0.45])

def predict_ct(file_bytes):
    img, original = preprocess_ct(file_bytes)

    x = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()[0]

    preds = (probs > thresholds).astype(int)

    return probs, preds, x, original
