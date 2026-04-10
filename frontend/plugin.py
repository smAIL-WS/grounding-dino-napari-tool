import napari
from magicgui import magicgui
from napari.qt.threading import thread_worker
from napari.layers import Image
import requests
import cv2
import numpy as np
import json
from pathlib import Path
import xml.etree.ElementTree as ET

BACKEND_URL = "http://localhost:8000" 
CACHE = {"bboxes": [], "labels": [], "scores": [], "layer_name": None, "image_shape": None}

def get_available_models(widget):
    try:
        return requests.get(f"{BACKEND_URL}/models").json().get("models", ["Error: Backend down"])
    except:
        return ["Backend unreachable"]

@magicgui(
    call_button="Run Detection",
    model_name={"choices": get_available_models},
    text_prompt={"label": "Prompt"},
    confidence_threshold={"widget_type": "FloatSlider", "min": 0.05, "max": 0.95, "step": 0.05, "label": "Confidence"},
    active_label={"choices": ["crop", "weed"], "label": "Drawing Label"}, # <--- NEW: Control manual drawing
    export_format={"choices": ["JSON", "PascalVOC"], "label": "Export Format"},
    download_button={"widget_type": "PushButton", "text": "Download Predictions"}
)
def detector_widget(
    viewer: napari.Viewer, 
    image_layer: Image, 
    model_name: str, 
    text_prompt: str = "crop . weed .", 
    confidence_threshold: float = 0.5,
    active_label: str = "crop",
    export_format: str = "JSON",
    download_button: bool = False 
):
    if image_layer is None: return

    img_data = image_layer.data
    CACHE["image_shape"] = img_data.shape
    
    if img_data.dtype != np.uint8:
        img_data = (img_data / img_data.max() * 255).astype(np.uint8)
    _, encoded_image = cv2.imencode('.png', img_data)

    def process_results(results):
        if "error" in results: return print(results["error"])
        CACHE["bboxes"] = results.get("bboxes", [])
        CACHE["labels"] = results.get("labels", [])
        CACHE["scores"] = results.get("scores", [])
        CACHE["layer_name"] = f"{image_layer.name}_detections"
        _on_slider_change(confidence_threshold)

    @thread_worker(connect={"returned": process_results})
    def run_inference():
        res = requests.post(
            f"{BACKEND_URL}/predict",
            data={"prompt": text_prompt, "model_name": model_name},
            files={"image": ("img.png", encoded_image.tobytes(), "image/png")}
        )
        res.raise_for_status()
        return res.json()

    run_inference()

# --- Helper Functions ---

def _on_download_click():
    # Logic remains the same, but now it includes manually drawn boxes too
    viewer = napari.current_viewer()
    layer_name = CACHE.get("layer_name")
    if not layer_name or layer_name not in viewer.layers:
        print("No detections layer found.")
        return
    
    layer = viewer.layers[layer_name]
    final_data = []
    
    # Export whatever is currently in the layer
    for i, bbox in enumerate(layer.data):
        label = layer.features['label'][i]
        # Bbox conversion logic
        y_coords = [p[0] for p in bbox]
        x_coords = [p[1] for p in bbox]
        final_data.append({
            "label": label,
            "bbox": [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        })

    # (File saving logic same as before...)
    fmt = detector_widget.export_format.value
    save_path = Path.home() / f"predictions_{fmt.lower()}.{ 'json' if fmt=='JSON' else 'xml' }"
    # ... [Keep JSON/XML saving block here] ...
    print(f"Exported {len(final_data)} boxes to {save_path}")

def _on_label_change(value: str):
    """Update Napari's drawing properties when you change the dropdown"""
    viewer = napari.current_viewer()
    layer_name = CACHE.get("layer_name")
    if layer_name and layer_name in viewer.layers:
        layer = viewer.layers[layer_name]
        # This tells Napari: "The next box I draw should have this label"
        layer.current_properties = {'label': [value]}
        # Update color for the next box
        color = "blue" if value == "crop" else "red"
        layer.current_edge_color = color

def _on_slider_change(value: float):
    if not CACHE["bboxes"]: return
    f_bboxes, f_labels, f_colors = [], [], []
    for i in range(len(CACHE["scores"])):
        if CACHE["scores"][i] >= value:
            label_text = CACHE["labels"][i].split(' ')[0]
            f_bboxes.append(CACHE["bboxes"][i])
            f_labels.append(label_text)
            f_colors.append("blue" if "crop" in label_text.lower() else "red")
            
    viewer = napari.current_viewer()
    layer_name = CACHE["layer_name"]
    
    if layer_name in viewer.layers:
        shape_layer = viewer.layers[layer_name]
        shape_layer.data = f_bboxes
        shape_layer.edge_color = f_colors
        shape_layer.features = {'label': f_labels}
    else:
        # Create layer with feature support
        viewer.add_shapes(
            f_bboxes, shape_type='polygon', edge_color=f_colors, face_color='transparent',
            edge_width=4, name=layer_name, 
            features={'label': f_labels},
            text={'text': '{label}', 'color': 'white'}
        )
    
    # Sync the initial drawing label
    _on_label_change(detector_widget.active_label.value)

# --- Signal Connections ---
detector_widget.download_button.changed.connect(_on_download_click)
detector_widget.confidence_threshold.changed.connect(_on_slider_change)
detector_widget.active_label.changed.connect(_on_label_change)

if __name__ == "__main__":
    v = napari.Viewer()
    v.window.add_dock_widget(detector_widget, area='right')
    napari.run()