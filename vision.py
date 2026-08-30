import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# --- 1. CUSTOM LABEL DICTIONARY ---
# Map the raw IDs your specific model spits out to the actual names.
# You will need to point the camera at pictures of these animals to find their IDs and add them here.
custom_labels = {
    17: “dog”,
20: “cow,
22: “bear”
18: “horse”,
0: “person”,
15: “bird”,
16: “cat”
}

TARGET_CLASSES = ["dog", "cat", "bird", "horse", "cow", "person”,”bear”] 

# --- 2. INITIALIZE TFLITE MODEL ---
print("Loading TinyML Model...")
interpreter = Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# --- 3. INITIALIZE CAMERA & TRACKING ---
cap = cv2.VideoCapture("http://10.202.205.203:4747/video")
print("System Ready. Continuously scanning video stream...")

# Temporal smoothing setup
consecutive_counts = {class_name: 0 for class_name in TARGET_CLASSES}
REQUIRED_CONSECUTIVE_FRAMES = 3  # Must see it 3 frames in a row to confirm

try:
    while True:
        if not cap.isOpened():
            print("[CAMERA] Stream lost. Reconnecting...")
            cap.open("http://10.202.205.203:4747/video")
            time.sleep(2)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05) # Prevent CPU lockup on dropped frames
            continue

        # Preprocessing (Convert BGR to RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to match model input dimensions (300x300)
        image_resized = cv2.resize(frame_rgb, (width, height))
        
        # Expand dimensions and ensure uint8 type for the quantized model
        input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

        # Run Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get Results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        # Track what was seen in THIS specific frame
        frame_detected_classes = set()

        for i in range(len(scores)):
            score = scores[i]
            if score > 0.40:  # Threshold for testing (adjust up to 0.60 later if needed)
                class_id = int(classes[i])
                
                # Look up ID in custom dictionary, default to "Unknown" if not mapped yet
                class_name = custom_labels.get(class_id, "Unknown")
                
                # Extract box coordinates for debugging [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = boxes[i]
                
                # Print everything it sees to help you map the IDs
                print(f"Raw ID: {class_id} | Mapped: {class_name} ({int(score * 100)}%) | Box -> Top: {ymin:.2f}, Left: {xmin:.2f}")

                if class_name in TARGET_CLASSES:
                    frame_detected_classes.add(class_name)

        # Update consecutive frame counters (Temporal Smoothing)
        for target in TARGET_CLASSES:
            if target in frame_detected_classes:
                consecutive_counts[target] += 1
                
                # Check if it has met the consecutive threshold requirement
                if consecutive_counts[target] >= REQUIRED_CONSECUTIVE_FRAMES:
                    print(f"====> [CONFIRMED MATCH] {target.upper()} verified across {REQUIRED_CONSECUTIVE_FRAMES} consecutive frames! <====")
                    # Reset counter for this target so it doesn't spam infinitely
                    consecutive_counts[target] = 0
            else:
                # Decay counter slightly if missing in a frame to prevent hanging states
                if consecutive_counts[target] > 0:
                    consecutive_counts[target] -= 1

except KeyboardInterrupt:
    print("\nExiting program...")
    cap.release()
