import cv2
import pickle
import numpy as np
import os

# --- CONFIGURATION ---
# Ensure this is the reference frame you extracted from your video
IMAGE_PATH = "reference_frame.jpg"  
PICKLE_FILE = "parking_slots.pkl"

# Load the image
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Erreur : impossible de charger l'image '{IMAGE_PATH}'")
    print("Assurez-vous d'avoir exécuté le script d'extraction d'image d'abord.")
    exit()

clone = image.copy()
points = []
parking_slots = []

# Load existing slots if the file exists
if os.path.exists(PICKLE_FILE):
    with open(PICKLE_FILE, "rb") as f:
        parking_slots = pickle.load(f)
    print(f"{len(parking_slots)} emplacements chargés.")
else:
    print(f"Nouveau fichier {PICKLE_FILE} sera créé.")

def redraw_all():
    """Refreshes the image display with all current slots and pending points."""
    global image
    image = clone.copy()
    for slot in parking_slots:
        pts = np.array(slot, np.int32)
        # Draw green boxes for finished slots
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Draw red dots for the points currently being clicked
    for p in points:
        cv2.circle(image, p, 4, (0, 0, 255), -1)

def auto_save():
    """Saves the current list of slots to the pickle file."""
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(parking_slots, f)
    print(f"Sauvegardé ({len(parking_slots)} emplacements au total)")

def mouse_callback(event, x, y, flags, param):
    global points, parking_slots

    # Left Click: Add a corner point
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 4:
            parking_slots.append([list(p) for p in points])
            points = []
            auto_save()
        redraw_all()

    # Right Click: Delete the last completed slot
    elif event == cv2.EVENT_RBUTTONDOWN:
        if parking_slots:
            parking_slots.pop()
            print("Dernier emplacement supprimé.")
            auto_save()
            redraw_all()

# --- WINDOW INITIALIZATION (THE FIX) ---
# 1. Create the window with WINDOW_NORMAL so it is resizable
cv2.namedWindow("Selector", cv2.WINDOW_NORMAL)

# 2. Resize the window to a large size on your screen
# You can change 1280, 720 to match your screen's resolution
cv2.resizeWindow("Selector", 1280, 720)

cv2.setMouseCallback("Selector", mouse_callback)

print("\n--- INSTRUCTIONS ---")
print("1. CLIC GAUCHE : Placez les 4 coins d'une place.")
print("2. CLIC DROIT  : Supprimez la dernière place enregistrée.")
print("3. TOUCHE 'R'  : Réinitialisez les points de la place en cours.")
print("4. TOUCHE 'Q'  : Quitter et fermer.\n")

# Initial draw
redraw_all()

while True:
    cv2.imshow("Selector", image)
    key = cv2.waitKey(1) & 0xFF
    
    # Quit
    if key == ord('q'):
        break
    
    # Reset current pending points
    if key == ord('r'):
        points = []
        redraw_all()
        print("Points en cours réinitialisés.")

cv2.destroyAllWindows()