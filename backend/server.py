import os
import functools
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from torchvision.ops import nms
from mmdet.apis import init_detector, inference_detector
import nltk
nltk.download([
    'punkt_tab',
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
], download_dir='/workspace/nltk_data')
nltk.data.path.append('/workspace/nltk_data')

app = FastAPI()

CONFIG_PATH = "/app/config.py"
CHECKPOINT_DIR = "/app/checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL_CACHE = {}

def load_model_cached(config, checkpoint, device):
    key = (config, checkpoint)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = init_detector(config, checkpoint, device=device)
    return _MODEL_CACHE[key]

# --- ADDED THIS SECTION TO FIX THE 404 ERROR ---
@app.get("/models")
async def list_models():
    """Returns a list of available .pth files to the Napari plugin"""
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Directory not found: {CHECKPOINT_DIR}")
        return {"models": []}
    
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    print(f"Available models found: {files}")
    return {"models": files}
# -----------------------------------------------

def run_sliding_window(model, img, text_prompt, patch_size=1000, overlap=200):
    h, w = img.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []
    step = patch_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end, x_end = min(y + patch_size, h), min(x + patch_size, w)
            patch = img[y:y_end, x:x_end]

            result = inference_detector(model, patch, text_prompt=text_prompt)
            pred = result.pred_instances
            
            if len(pred.bboxes) > 0:
                boxes = pred.bboxes.cpu().numpy()
                boxes[:, [0, 2]] += x 
                boxes[:, [1, 3]] += y 
                all_boxes.append(torch.tensor(boxes))
                all_scores.append(pred.scores.cpu())
                all_labels.append(pred.labels.cpu())

    if not all_boxes:
        return np.array([]), np.array([]), np.array([])

    combined_boxes = torch.cat(all_boxes)
    combined_scores = torch.cat(all_scores)
    combined_labels = torch.cat(all_labels)
    keep = nms(combined_boxes, combined_scores, iou_threshold=0.3)

    return combined_boxes[keep].numpy(), combined_scores[keep].numpy(), combined_labels[keep].numpy()

@app.post("/predict")
async def predict(image: UploadFile = File(...), prompt: str = Form(...), model_name: str = Form(...)):
    contents = await image.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    ckpt = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.exists(ckpt): return {"error": f"Model {model_name} not found"}
    
    model = load_model_cached(CONFIG_PATH, ckpt, DEVICE)
    boxes, scores, labels = run_sliding_window(model, img, prompt)

    napari_boxes, labels_out, raw_scores = [], [], []
    prompt_classes = [c.strip() for c in prompt.split('.') if c.strip()]

    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = [float(v) for v in boxes[i]]
        napari_boxes.append([[y_min, x_min], [y_max, x_min], [y_max, x_max], [y_min, x_max]])
        
        lbl_idx = int(labels[i])
        lbl_str = prompt_classes[lbl_idx] if lbl_idx < len(prompt_classes) else f"id_{lbl_idx}"
        
        labels_out.append(f"{lbl_str} ({scores[i]:.2f})")
        raw_scores.append(float(scores[i]))

    return {"bboxes": napari_boxes, "labels": labels_out, "scores": raw_scores}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)