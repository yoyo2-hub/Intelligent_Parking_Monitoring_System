import cv2
import pickle
import numpy as np
import os
from ultralytics import YOLO
from shapely.geometry import Polygon
from flask import Flask, Response, render_template_string, jsonify
from flask_cors import CORS
import threading
import webbrowser
import time

# --- CONFIGURATION ---
VIDEO_PATH = "My movie 3.mp4"
PICKLE_FILE = "parking_slots.pkl"
MODEL_PATH = "yolo11s.pt"

FRAME_SKIP = 3          # YOLO toutes les 3 frames
OCCUPANCY_THRESHOLD = 45
REQUIRED_STABILITY_FRAMES = 15

# --- FLASK APP ---
app = Flask(__name__)
CORS(app)

# Variables globales
current_frame = None
frame_lock = threading.Lock()
parking_stats = {
    "total": 0,
    "occupied": 0,
    "progress": 0,
    "available": 0
}

# --- CHARGEMENT HTML ---
with open('final.html', 'r', encoding='utf-8') as f:
    HTML_TEMPLATE = f.read()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if current_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(parking_stats)

# --- INITIALISATION ---
if not os.path.exists(PICKLE_FILE):
    raise Exception(f"File {PICKLE_FILE} not found!")

with open(PICKLE_FILE, "rb") as f:
    parking_slots = pickle.load(f)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# ORB pour stabilisation
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, first_frame = cap.read()
if not ret: exit()

gray_ref = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
last_valid_matrix = np.eye(3)

# --- STATE MACHINE DATA ---
slot_states = {i: 0 for i in range(len(parking_slots))}
stability_counters = {i: 0 for i in range(len(parking_slots))}

# --- LANCEMENT FLASK + NAVIGATEUR ---
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

time.sleep(2)
webbrowser.open('http://localhost:5000')

print("🚀 Serveur démarré sur http://localhost:5000")
print("📹 L'interface web s'ouvre automatiquement...")
print("⏹️ Appuyez sur 'q' dans la fenêtre vidéo pour arrêter")

# --- MAIN LOOP ---
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0: continue

    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)
    if des_curr is not None:
        matches = sorted(bf.match(des_ref, des_curr), key=lambda x: x.distance)
        if len(matches) > 15:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if matrix is not None: last_valid_matrix = matrix

    # --- AI DETECTION ---
    results = model.predict(frame, classes=[2, 3, 5, 7], verbose=False, conf=0.5)
    result = results[0]
    detections = result.masks.xy if result.masks is not None else result.boxes.xyxy.cpu().numpy()

    annotated_frame = frame.copy()
    occ_count, prog_count = 0, 0

    # --- STATE MACHINE ---
    for idx, slot in enumerate(parking_slots):
        slot_pts = np.array(slot, np.float32).reshape(-1, 1, 2)
        trans_pts = cv2.perspectiveTransform(slot_pts, last_valid_matrix).reshape(-1, 2)
        slot_poly = Polygon(trans_pts)

        max_inter = 0
        for det in detections:
            car_poly = Polygon(det) if result.masks is not None else Polygon([
                (det[0], det[1]), (det[2], det[1]), (det[2], det[3]), (det[0], det[3])
            ])
            if slot_poly.intersects(car_poly):
                area = slot_poly.intersection(car_poly).area
                if area > max_inter: max_inter = area

        raw_occ = (max_inter / slot_poly.area) * 100

        # --- STATE MACHINE LOGIC ---
        if slot_states[idx] == 0:  # FREE
            if raw_occ > OCCUPANCY_THRESHOLD:
                slot_states[idx] = 1
                stability_counters[idx] = 1
            color = (0, 255, 0)

        elif slot_states[idx] == 1:  # PROGRESS
            if raw_occ > OCCUPANCY_THRESHOLD:
                stability_counters[idx] += 1
                if stability_counters[idx] >= REQUIRED_STABILITY_FRAMES:
                    slot_states[idx] = 2
            else:
                slot_states[idx] = 0
                stability_counters[idx] = 0
            color = (0, 255, 255)
            prog_count += 1

        elif slot_states[idx] == 2:  # OCCUPIED
            if raw_occ < 20:
                slot_states[idx] = 0
                stability_counters[idx] = 0
            color = (0, 0, 255)
            occ_count += 1

        center_x = int(np.mean(trans_pts[:, 0]))
        center_y = int(np.mean(trans_pts[:, 1]))
        cv2.circle(annotated_frame, (center_x, center_y), 11, color, -1)
        cv2.putText(annotated_frame, f"{int(raw_occ)}%", (center_x + 15, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # --- UPDATE DASHBOARD STATS ---
    available_count = len(parking_slots) - occ_count - prog_count
    parking_stats["total"] = len(parking_slots)
    parking_stats["occupied"] = occ_count
    parking_stats["progress"] = prog_count
    parking_stats["available"] = available_count

    # --- DISPLAY DASHBOARD ON FRAME ---
    box_x1, box_y1, box_w, box_h = 20, 30, 380, 200
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x1 + box_w, box_y1 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)
    cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x1 + box_w, box_y1 + box_h), (200, 200, 200), 2)

    def draw_text(txt, y_pos, size, color, thick):
        text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, size, thick)[0]
        text_x = box_x1 + (box_w - text_size[0]) // 2
        cv2.putText(annotated_frame, txt, (text_x, box_y1 + y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

    draw_text(f"TOTAL SPOTS: {len(parking_slots)}", 45, 0.8, (255, 255, 255), 2)
    draw_text(f"OCCUPIED: {occ_count}", 90, 0.9, (0, 0, 255), 3)
    draw_text(f"PROGRESS: {prog_count}", 135, 0.7, (0, 255, 255), 2)
    draw_text(f"AVAILABLE: {available_count}", 185, 0.9, (0, 255, 0), 3)

    # --- UPDATE FLASK FRAME ---
    with frame_lock:
        current_frame = annotated_frame.copy()

    cv2.imshow("Parking Monitor", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
