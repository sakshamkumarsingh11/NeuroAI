
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from predict import predict_ct, model, classes
from gradcam import generate_heatmap, overlay_heatmap
from utils import encode_image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        probs, preds, tensor, original = predict_ct(contents)

        heatmap = generate_heatmap(model, tensor)
        overlay = overlay_heatmap(original, heatmap)

        # --- Medically correct metrics ---

        tbi_detected = bool(preds.sum() > 0)

        # Confidence: max probability across all classes (as percentage)
        confidence = float(np.max(probs)) * 100
        if np.isnan(confidence):
            confidence = 0.0

        # TBI probability = max class probability
        tbi_prob = float(np.max(probs)) * 100
        if np.isnan(tbi_prob):
            tbi_prob = 0.0
        no_tbi_prob = round(100 - tbi_prob, 2)

        # Detected hemorrhage types
        detected_types = [c for c, p in zip(classes, preds) if p == 1]

        # Risk level based on max probability
        max_prob = float(np.max(probs))
        if max_prob > 0.85:
            risk = "HIGH"
        elif max_prob > 0.6:
            risk = "MODERATE"
        else:
            risk = "LOW"

        # Estimated affected region from Grad-CAM activation area
        # (percentage of heatmap pixels above a threshold, NOT real segmentation)
        cam_active = float(np.sum(heatmap > 0.3)) / (heatmap.shape[0] * heatmap.shape[1]) * 100
        estimated_region_pct = round(cam_active, 1) if tbi_detected else 0.0

        # Status text
        if tbi_detected:
            status = "Intracranial hemorrhage detected"
        else:
            status = "No intracranial hemorrhage detected"

        # Per-class probabilities for detailed view
        class_probabilities = {c: round(float(p) * 100, 2) for c, p in zip(classes, probs)}

        return JSONResponse(content={
            "status": status,
            "tbi_detected": tbi_detected,
            "detected_types": detected_types,
            "confidence": round(confidence, 2),
            "risk": risk,
            "tbi_probability": round(tbi_prob, 2),
            "no_tbi_probability": round(no_tbi_prob, 2),
            "estimated_region_pct": estimated_region_pct,
            "class_probabilities": class_probabilities,
            "heatmap": encode_image(overlay),
            "note": "Highlighted region shows model attention (Grad-CAM), not exact lesion boundary"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
