import cv2
import pickle
import numpy as np
import os
from ultralytics import YOLO
from shapely.geometry import Polygon

# --- CONFIGURATION ---
VIDEO_PATH = "video.mp4"
PICKLE_FILE = "parking_slots.pkl"
OUTPUT_VIDEO = "parking_pro_stable.mp4"
MODEL_PATH = "yolo11s.pt"

FRAME_SKIP = 3        
OCCUPANCY_THRESHOLD = 45 
CONFIRMED_THRESHOLD = 50 
REQUIRED_STABILITY_FRAMES = 15 

# --- INITIALIZATION ---
if not os.path.exists(PICKLE_FILE):
    raise Exception(f"File {PICKLE_FILE} not found!")

with open(PICKLE_FILE, "rb") as f:
    parking_slots = pickle.load(f)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, first_frame = cap.read()
if not ret: exit()

gray_ref = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
last_valid_matrix = np.eye(3)

width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'avc1'), int(cap.get(5))//FRAME_SKIP, (width, height))

# --- STATE MACHINE DATA ---
slot_states = {i: 0 for i in range(len(parking_slots))}
stability_counters = {i: 0 for i in range(len(parking_slots))}

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % FRAME_SKIP != 0: continue

    # 1. CAMERA STABILIZATION
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)
    if des_curr is not None:
        matches = sorted(bf.match(des_ref, des_curr), key=lambda x: x.distance)
        if len(matches) > 15:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if matrix is not None: last_valid_matrix = matrix
    
    # 2. AI DETECTION
    results = model.predict(frame, classes=[2, 3, 5, 7], verbose=False, conf=0.5)
    result = results[0]
    detections = result.masks.xy if result.masks is not None else result.boxes.xyxy.cpu().numpy()

    annotated_frame = frame.copy()
    occ_count, prog_count = 0, 0

    # 3. STATE MACHINE LOGIC
    for idx, slot in enumerate(parking_slots):
        slot_pts = np.array(slot, np.float32).reshape(-1, 1, 2)
        trans_pts = cv2.perspectiveTransform(slot_pts, last_valid_matrix).reshape(-1, 2)
        slot_poly = Polygon(trans_pts)
        
        max_inter = 0
        for det in detections:
            car_poly = Polygon(det) if result.masks is not None else Polygon([(det[0], det[1]), (det[2], det[1]), (det[2], det[3]), (det[0], det[3])])
            if slot_poly.intersects(car_poly):
                area = slot_poly.intersection(car_poly).area
                if area > max_inter: max_inter = area

        raw_occ = (max_inter / slot_poly.area) * 100

        # --- THE MACHINE ---
        if slot_states[idx] == 0: # FREE
            if raw_occ > OCCUPANCY_THRESHOLD:
                slot_states[idx] = 1
                stability_counters[idx] = 1
            color = (0, 255, 0) # Green

        elif slot_states[idx] == 1: # PROGRESS
            if raw_occ > OCCUPANCY_THRESHOLD:
                stability_counters[idx] += 1
                if stability_counters[idx] >= REQUIRED_STABILITY_FRAMES:
                    slot_states[idx] = 2
            else:
                slot_states[idx] = 0
                stability_counters[idx] = 0
            color = (0, 255, 255) # Yellow
            prog_count += 1

        elif slot_states[idx] == 2: # OCCUPIED
            if raw_occ < 20: 
                slot_states[idx] = 0
                stability_counters[idx] = 0
            color = (0, 0, 255) # Red
            occ_count += 1

        # --- VISUALS (Centrées) ---
        # Calcul du centre géométrique du polygone pour placer la bulle
        center_x = int(np.mean(trans_pts[:, 0]))
        center_y = int(np.mean(trans_pts[:, 1]))
        
        # Bulle centrale
        cv2.circle(annotated_frame, (center_x, center_y), 11, color, -1)
        # Texte à côté de la bulle
        cv2.putText(annotated_frame, f"{int(raw_occ)}%", (center_x + 15, center_y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # --- 4. DASHBOARD STYLÉ ---
    box_x1, box_y1, box_w, box_h = 20, 30, 380, 200
    overlay = annotated_frame.copy()
    # Rectangle noir arrondi (simulation par épaisseur)
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x1 + box_w, box_y1 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)
    # Bordure blanche élégante
    cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x1 + box_w, box_y1 + box_h), (200, 200, 200), 2)
    
    def draw_text(txt, y_pos, size, color, thick):
        # Centrage automatique du texte dans le dashboard
        text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0]
        text_x = box_x1 + (box_w - text_size[0]) // 2
        cv2.putText(annotated_frame, txt, (text_x, box_y1 + y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

    draw_text(f"TOTAL SPOTS: {len(parking_slots)}", 45, 0.8, (255, 255, 255), 2)
    draw_text(f"OCCUPIED: {occ_count}", 90, 0.9, (0, 0, 255), 3)
    draw_text(f"PROGRESS: {prog_count}", 135, 0.7, (0, 255, 255), 2)
    draw_text(f"AVAILABLE: {len(parking_slots) - occ_count}", 185, 0.9, (0, 255, 0), 3)
    
    out.write(annotated_frame)
    cv2.imshow("Stable Monitor Pro", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()