import cv2
import numpy as np
import time
import os
import random
import pygame
import requests
from datetime import datetime
from gpiozero import MotionSensor
from tflite_runtime.interpreter import Interpreter

# ==========================================
# 1. TELEGRAM API SETUP
# ==========================================
TELEGRAM_BOT_TOKEN = "LOL"  
TELEGRAM_CHAT_ID = "Not this time"      

def send_telegram_alert(animal_name):
    """Sends a formatted message to Telegram with the animal name and timestamp."""
    try:
        current_time = datetime.now().strftime("%I:%M:%S %p on %d %b %Y")
        message = f"🚨 ALERT: {animal_name.upper()} detected!\n🕒 Time: {current_time}"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        
        # Runs request with a timeout so a bad connection doesn't freeze the camera loop
        response = requests.post(url, json=payload, timeout=3)
        if response.status_code == 200:
            print(f"[TELEGRAM] Alert sent for {animal_name}.")
        else:
            print(f"[TELEGRAM ERROR] Code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[TELEGRAM ERROR] Failed to send message: {e}")

# ==========================================
# 2. AUDIO DETERRENT SETUP
# ==========================================
pygame.mixer.init()
SOUNDS_FOLDER = "/home/pi/AudioD"  

def play_random_deterrent():
    """Picks a random .mp3 file from the AudioD folder and plays it."""
    try:
        if not os.path.exists(SOUNDS_FOLDER):
            print(f"[AUDIO ERROR] Folder '{SOUNDS_FOLDER}' not found.")
            return

        audio_files = [f for f in os.listdir(SOUNDS_FOLDER) if f.endswith('.mp3')]
        if not audio_files:
            return
            
        sound_file = random.choice(audio_files)
        sound_path = os.path.join(SOUNDS_FOLDER, sound_file)
        
        print(f"[AUDIO] Playing deterrent: {sound_file}")
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()
        
    except Exception as e:
        print(f"[AUDIO ERROR] Could not play sound: {e}")

# ==========================================
# 3. VISION MODEL SETUP
# ==========================================
custom_labels = {
    74: "dog",      
    75: "dog",      
    # 123: "person",
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
# 4. VISION PIPELINE
# ==========================================
def run_vision_pipeline():
    """Runs the AI camera for 10 seconds to check what triggered the motion."""
    print("[VISION] Camera active. Scanning for targets...")
    consecutive_counts = {class_name: 0 for class_name in TARGET_CLASSES}
    
    # Track which animals we already alerted Telegram about in this 10-second window
    already_alerted = set()
    
    start_time = time.time()
    
    while time.time() - start_time < 10:
        if not cap.isOpened():
            cap.open("http://10.202.205.203:4747/video")
            time.sleep(1)
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

        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        frame_detected_classes = set()

        for i in range(len(scores)):
            score = scores[i]
            if score > 0.40:
                class_id = int(classes[i])
                class_name = custom_labels.get(class_id, "Unknown")
                
                if class_name in TARGET_CLASSES:
                    frame_detected_classes.add(class_name)

        for target in TARGET_CLASSES:
            if target in frame_detected_classes:
                consecutive_counts[target] += 1
                
                if consecutive_counts[target] >= REQUIRED_CONSECUTIVE_FRAMES:
                    print(f"====> [CONFIRMED MATCH] {target.upper()} DETECTED! <====")
                    
                    # 1. Send Telegram Alert (Only once per animal per 10-second session)
                    if target not in already_alerted:
                        send_telegram_alert(target)
                        already_alerted.add(target)
                    
                    # 2. Play Audio Deterrent
                    if not pygame.mixer.music.get_busy():
                        play_random_deterrent()
                        
                    consecutive_counts[target] = 0
            else:
                if consecutive_counts[target] > 0:
                    consecutive_counts[target] -= 1
                    
    print("[VISION] Scan window ended. Going back to sleep...")

# ==========================================
# 5. PIR SENSOR SETUP & MAIN LOOP
# ==========================================
pir = MotionSensor(4, queue_len=1, threshold=0.1)
last_trigger_time = 0
COOL_DOWN_TIME = 15  

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

pir.when_motion = on_motion_detected
pir.when_no_motion = off_motion_detected

print("\n==================================================")
print("SYSTEM INITIALIZED: Monitoring boundary for wildlife")
print("==================================================\n")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nExiting program...")
    cap.release()
    pygame.quit()
