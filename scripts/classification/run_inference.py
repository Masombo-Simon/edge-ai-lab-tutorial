import os
import cv2
import yaml
import json
import numpy as np
import onnxruntime as ort

# ==========================================
# 1. LOAD CONFIGURATION
# ==========================================
config_path = "mobilenet_v2_tv_qat-p2_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

preprocess_opts = config.get('preprocessing', {})
crop_size = preprocess_opts.get('crop_size', [224, 224]) 
mean_values = preprocess_opts.get('mean', [0.485, 0.456, 0.406])
std_values = preprocess_opts.get('std', [0.229, 0.224, 0.225])

# Load your local 1000-class JSON dictionary if you downloaded it
# Fallback to empty dict if you haven't downloaded it yet
try:
    with open("imagenet_classes.json", "r") as f:
        labels_map = json.load(f)
except FileNotFoundError:
    labels_map = {}

# ==========================================
# 2. SET UP DATASET PATH
# ==========================================
# Point this to your folder of images
dataset_dir = "my_dataset" 

if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Could not find the directory: {dataset_dir}")

# List all image files in that directory
supported_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(supported_extensions)]

print(f"Found {len(image_files)} images in '{dataset_dir}' folder.\n")

# ==========================================
# 3. INITIALIZE ONNX MODEL
# ==========================================
model_path = "mobilenet_v2_tv_qat-p2.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# ==========================================
# 4. LOOP AND RUN INFERENCE
# ==========================================
for filename in image_files:
    full_image_path = os.path.join(dataset_dir, filename)
    
    # Preprocess image
    raw_img = cv2.imread(full_image_path)
    if raw_img is None:
        continue
        
    resized_img = cv2.resize(raw_img, (crop_size[1], crop_size[0]))
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose(2, 0, 1)
    img_floats = img_chw.astype(np.float32) / 255.0

    mean = np.array(mean_values).reshape(3, 1, 1)
    std = np.array(std_values).reshape(3, 1, 1)
    normalized_img = ((img_floats - mean) / std).astype(np.float32)
    input_tensor = np.expand_dims(normalized_img, axis=0)

    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    
    # Parse prediction
    predictions = outputs[0][0] 
    predicted_class_id = np.argmax(predictions) 
    class_name = labels_map.get(str(predicted_class_id), f"Class ID {predicted_class_id}")

    print(f"File: {filename} ➔ Identified As: {class_name}")