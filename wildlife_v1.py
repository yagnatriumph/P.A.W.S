import cv2
import numpy as np
import time
import os
import random
import pygame
from gpiozero import MotionSensor
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1. AUDIO DETERRENT SETUP
# ==========================================
pygame.mixer.init()
# Pointing to the folder you uploaded via SCP
SOUNDS_FOLDER = "/home/pi/AudioD"  

def play_random_deterrent():
    """Picks a random .mp3 file from the AudioD folder and plays it."""
    try:
        if not os.path.exists(SOUNDS_FOLDER):
            print(f"[AUDIO ERROR] Folder '{SOUNDS_FOLDER}' not found.")
            return

        # Read only .mp3 files based on your scp logs
        audio_files = [f for f in os.listdir(SOUNDS_FOLDER) if f.endswith('.mp3')]
        
        if not audio_files:
            print(f"[AUDIO ERROR] No .mp3 files found in '{SOUNDS_FOLDER}'.")
            return
            
        sound_file = random.choice(audio_files)
        sound_path = os.path.join(SOUNDS_FOLDER, sound_file)
        
        print(f"[AUDIO] Playing deterrent: {sound_file}")
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        
    except Exception as e:
        print(f"[AUDIO ERROR] Could not play sound: {e}")


# ==========================================
# 2. VISION MODEL SETUP
# ==========================================
# Update these IDs with the raw IDs your model outputs for these animals!
custom_labels = {
    17: “dog”,
20: “cow,
22: “bear”,
18: “horse”,
0: “person”,
15: “bird”,
16: “cat”
}
TARGET_CLASSES = ["dog", "cat", "bird", "horse", "cow", "person"] 
REQUIRED_CONSECUTIVE_FRAMES = 3  

print("Loading TinyML Model...")
interpreter = Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

cap = cv2.VideoCapture("http://10.202.205.203:4747/video")


# ==========================================
# 3. VISION PIPELINE (WAKES UP ON MOTION)
# ==========================================
def run_vision_pipeline():
    """Runs the AI camera for 10 seconds to check what triggered the motion."""
    print("[VISION] Camera active. Scanning for targets...")
    consecutive_counts = {class_name: 0 for class_name in TARGET_CLASSES}
    start_time = time.time()
    
    # Keep scanning for 10 seconds after the PIR sensor is triggered
    while time.time() - start_time < 10:
        if not cap.isOpened():
            cap.open("http://10.202.205.203:4747/video")
            time.sleep(1)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        # Prepare image for TFLite
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

        # Run AI Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        frame_detected_classes = set()

        # Check what is in the current frame
        for i in range(len(scores)):
            score = scores[i]
            if score > 0.40:
                class_id = int(classes[i])
                class_name = custom_labels.get(class_id, "Unknown")
                
                if class_name in TARGET_CLASSES:
                    frame_detected_classes.add(class_name)

        # Update consecutive frame counters (Temporal Smoothing)
        for target in TARGET_CLASSES:
            if target in frame_detected_classes:
                consecutive_counts[target] += 1
                
                if consecutive_counts[target] >= REQUIRED_CONSECUTIVE_FRAMES:
                    print(f"====> [CONFIRMED MATCH] {target.upper()} DETECTED! <====")
                    
                    # Play sound ONLY if a sound isn't already playing
                    if not pygame.mixer.music.get_busy():
                        play_random_deterrent()
                        
                    # Reset counter so it doesn't spam alerts instantly
                    consecutive_counts[target] = 0
            else:
                if consecutive_counts[target] > 0:
                    consecutive_counts[target] -= 1
                    
    print("[VISION] Scan window ended. Going back to sleep...")


# ==========================================
# 4. PIR SENSOR SETUP & MAIN LOOP
# ==========================================
pir = MotionSensor(4, queue_len=1, threshold=0.1)
last_trigger_time = 0
COOL_DOWN_TIME = 15  # Wait 15 seconds before allowing another trigger

def on_motion_detected():
    global last_trigger_time
    current_time = time.time()
    
    if current_time - last_trigger_time > COOL_DOWN_TIME:
        print("\n[PIR EVENT] Motion detected! Waking up AI vision pipeline...")
        last_trigger_time = current_time
        run_vision_pipeline()
    else:
        print("[PIR EVENT] Motion ignored (System is in Cooldown).")

def off_motion_detected():
    print("[PIR EVENT] Area is clear. Sensor ready.")

# Assign PIR sensor events
pir.when_motion = on_motion_detected
pir.when_no_motion = off_motion_detected

print("\n==================================================")
print("SYSTEM INITIALIZED: Monitoring boundary for wildlife")
print("==================================================\n")

try:
    # This loop keeps the script running endlessly in the background
    # while the PIR sensor waits for physical movement to trigger the camera
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nExiting program...")
    cap.release()
    pygame.quit()
