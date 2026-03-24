import cv2

# --- CONFIGURATION ---
VIDEO_PATH = "video.mp4"
OUTPUT_NAME = "reference_frame.jpg"

def extract_first_frame():
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    # Read the very first frame
    success, frame = cap.read()
    
    if success:
        cv2.imwrite(OUTPUT_NAME, frame)
        print(f"Success! {OUTPUT_NAME} has been saved.")
        print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("Error: Could not read the first frame.")
    
    cap.release()

if __name__ == "__main__":
    extract_first_frame()