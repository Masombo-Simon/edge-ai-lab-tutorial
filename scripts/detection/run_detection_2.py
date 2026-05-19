import os
import cv2
import yaml
import urllib.request
import numpy as np
import ai_edge_litert.interpreter as litert

# ==========================================
# 1. LOAD OBJECT DETECTION CONFIG (.yaml)
# ==========================================
config_path = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8_config.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

preprocess_opts = config.get('preprocessing', {})
crop_size = preprocess_opts.get('crop_size', [320, 320]) 
mean_values = preprocess_opts.get('mean', [127.5, 127.5, 127.5])
std_values = preprocess_opts.get('std', [127.5, 127.5, 127.5])

print("--- Config Loaded ---")
print(f"Target Input Resolution: {crop_size[1]}x{crop_size[0]}")

# ==========================================
# 2. FETCH THE COCO DATASET LABELS 
# ==========================================
COCO_LABELS_URL = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
try:
    req = urllib.request.Request(COCO_LABELS_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as response:
        coco_classes = [line.strip().decode('utf-8') for line in response.readlines()]
    print("Success! COCO dataset labels loaded online.\n")
except Exception:
    print("Failed to download labels. Defaulting to generic IDs.\n")
    coco_classes = [f"Class {i}" for i in range(100)]

# ==========================================
# 3. INITIALIZE TFLITE/LITERT ENGINE
# ==========================================
model_path = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Could not find downloaded file: {model_path}")

interpreter = litert.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# 4. SET UP DATASET PATH
# ==========================================
dataset_dir = "my_detection_dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Created empty folder '{dataset_dir}'. Add some images to it and rerun!")

supported_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(supported_extensions)]

if not image_files:
    print(f"Please place test images inside the '{dataset_dir}' folder first.")
    exit()

print(f"Found {len(image_files)} test images. Starting detection...")

# ==========================================
# 5. LOOP AND RUN DETECTION INFERENCE
# ==========================================
for filename in image_files:
    full_image_path = os.path.join(dataset_dir, filename)
    raw_img = cv2.imread(full_image_path)
    if raw_img is None:
        continue
        
    orig_h, orig_w = raw_img.shape[:2]

    # Preprocessing
    resized_img = cv2.resize(raw_img, (crop_size[1], crop_size[0]))
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_chw = img_rgb.transpose(2, 0, 1)
    img_floats = img_chw.astype(np.float32)

    mean = np.array(mean_values).reshape(3, 1, 1)
    std = np.array(std_values).reshape(3, 1, 1)
    normalized_img = ((img_floats - mean) / std).astype(np.float32)
    input_tensor = np.expand_dims(normalized_img, axis=0)

    # TFLite networks generally require Channels-Last format (NHWC) 
    # instead of Channels-First (NCHW). Let's transform it if required:
    if input_details[0]['shape'][-1] == 3: 
        input_tensor = input_tensor.transpose(0, 2, 3, 1)

    # Execute Engine
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # Parse standard TFLite detection outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])

    print(f"\n--- File: {filename} ---")
    
    for i in range(num_detections):
        score = scores[i]
        if score < 0.45: 
            continue
            
        class_id = int(class_ids[i])
        box = boxes[i]
        label = coco_classes[class_id] if class_id < len(coco_classes) else f"ID {class_id}"
        
        # TFLite box format: [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        left = int(xmin * orig_w)
        top = int(ymin * orig_h)
        right = int(xmax * orig_w)
        bottom = int(ymax * orig_h)

        cv2.rectangle(raw_img, (left, top), (right, bottom), (0, 255, 0), 2)
        display_text = f"{label}: {score*100:.1f}%"
        cv2.putText(raw_img, display_text, (left, max(top - 10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"   [Found]: {label} ({score*100:.1f}%)")

    output_filename = f"detected_{filename}"
    cv2.imwrite(output_filename, raw_img)
    print(f"   Saved marked visualization ➔ {output_filename}")