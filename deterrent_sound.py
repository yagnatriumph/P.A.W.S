import cv2
import numpy as np
import time
import os
import random
import pygame
from tflite_runtime.interpreter import Interpreter

# --- 1. AUDIO DETERRENT SETUP ---
# Initialize Pygame Mixer
pygame.mixer.init()
SOUNDS_FOLDER = "sounds"  # The folder where you put your .wav files

def play_random_deterrent():
    """Picks a random .wav file from the sounds folder and plays it."""
    try:
        if not os.path.exists(SOUNDS_FOLDER):
            print(f"[AUDIO] Folder '{SOUNDS_FOLDER}' not found.")
            return

        wav_files = [f for f in os.listdir(SOUNDS_FOLDER) if f.endswith('.wav')]
        
        if not wav_files:
            print(f"[AUDIO] No .wav files found in '{SOUNDS_FOLDER}'.")
            return
            
        # Pick a random sound
        sound_file = random.choice(wav_files)
        sound_path = os.path.join(SOUNDS_FOLDER, sound_file)
        
        print(f"[AUDIO] Playing deterrent: {sound_file}")
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        
    except Exception as e:
        print(f"[AUDIO ERROR] Could not play sound: {e}")

# --- 2. CUSTOM LABEL DICTIONARY ---
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

# --- 3. INITIALIZE TFLITE MODEL ---
print("Loading TinyML Model...")
interpreter = Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# --- 4. INITIALIZE CAMERA & TRACKING ---
cap = cv2.VideoCapture("http://10.202.205.203:4747/video")
print("System Ready. Continuously scanning video stream...")

consecutive_counts = {class_name: 0 for class_name in TARGET_CLASSES}
REQUIRED_CONSECUTIVE_FRAMES = 3  

try:
    while True:
        if not cap.isOpened():
            print("[CAMERA] Stream lost. Reconnecting...")
            cap.open("http://10.202.205.203:4747/video")
            time.sleep(2)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05) 
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        frame_detected_classes = set()

        for i in range(len(scores)):
            score = scores[i]
            if score > 0.40:  
                class_id = int(classes[i])
                class_name = custom_labels.get(class_id, "Unknown")
                ymin, xmin, ymax, xmax = boxes[i]
                
                print(f"Raw ID: {class_id} | Mapped: {class_name} ({int(score * 100)}%) | Box -> Top: {ymin:.2f}, Left: {xmin:.2f}")

                if class_name in TARGET_CLASSES:
                    frame_detected_classes.add(class_name)

        # Update consecutive frame counters
        for target in TARGET_CLASSES:
            if target in frame_detected_classes:
                consecutive_counts[target] += 1
                
                if consecutive_counts[target] >= REQUIRED_CONSECUTIVE_FRAMES:
                    print(f"====> [CONFIRMED MATCH] {target.upper()} verified across {REQUIRED_CONSECUTIVE_FRAMES} frames! <====")
                    
                    # TRIGGER AUDIO ONLY IF NOTHING IS CURRENTLY PLAYING
                    if not pygame.mixer.music.get_busy():
                        play_random_deterrent()
                        
                    consecutive_counts[target] = 0
            else:
                if consecutive_counts[target] > 0:
                    consecutive_counts[target] -= 1

except KeyboardInterrupt:
    print("\nExiting program...")
    cap.release()
    pygame.quit()
