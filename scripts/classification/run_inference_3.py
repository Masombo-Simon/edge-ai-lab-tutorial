import os
import cv2
import yaml
import json
import ssl
import urllib.request
import numpy as np
import onnxruntime as ort

# Bypass SSL verification if Windows/Python environment is throwing security flags
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# 1. LOAD CONFIGURATION
# ==========================================
config_path = "regnet_x_800mf_tv_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

preprocess_opts = config.get('preprocessing', {})
crop_size = preprocess_opts.get('crop_size', [224, 224]) 
mean_values = preprocess_opts.get('mean', [0.485, 0.456, 0.406])
std_values = preprocess_opts.get('std', [0.229, 0.224, 0.225])

# ==========================================
# 2. FETCH LABELS FROM THE ONLINE LINK
# ==========================================
# Official, highly stable TensorFlow ImageNet stable URL
ORIGINAL_LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

print("Connecting to storage registry to download the ImageNet label dataset...")
req = urllib.request.Request(ORIGINAL_LABELS_URL, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=10) as response:
    raw_data = json.loads(response.read().decode())
    # TensorFlow format is {"0": ["n01440764", "tench"]}. We extract just the clean name.
    labels_map = {key: value[1].replace('_', ' ').title() for key, value in raw_data.items()}
print("Success! Dataset labels fetched from the cloud.\n")
# ==========================================
# 3. SET UP DATASET PATH & MODEL
# ==========================================
dataset_dir = "my_dataset" 
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Could not find the directory: {dataset_dir}")

supported_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(supported_extensions)]

print(f"Found {len(image_files)} images in '{dataset_dir}' folder.")

model_path = "regnet_x_800mf_tv.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

print("Processing images...\n")

# ==========================================
# 4. LOOP AND RUN INFERENCE
# ==========================================
for filename in image_files:
    full_image_path = os.path.join(dataset_dir, filename)
    
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

    outputs = session.run(None, {input_name: input_tensor})
    
    predictions = outputs[0][0] 
    predicted_class_id = np.argmax(predictions) 
    
    # Map class ID directly using the downloaded dataset map
    class_name = labels_map.get(str(predicted_class_id), f"Class ID {predicted_class_id}")

    print(f"File: {filename:<18} ➔ Identified As: {class_name}")