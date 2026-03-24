# **Intelligent_Parking_Monitoring_System (Vision-AI)**

# **Project Overview**
Searching for parking takes an average of 5 to 20 minutes per trip, increasing driver stress and CO₂ emissions. Our system provides a high-tech answer:
Automated occupancy detection via advanced instance segmentation.
Enhanced reliability under degraded conditions (night, rain, vibrations).
Interactive Dashboard for centralized data management and real-time insights.

# **Key Features**
Multi-Vehicle Detection: Accurately identifies and classifies cars, motorcycles, buses, and heavy-duty trucks.
Instance Segmentation (YOLOv11): Uses precise masks that fit the vehicle's body. These masks are processed via the Shapely algorithm to calculate exact intersection areas, avoiding the errors common with traditional bounding boxes.
Spatial Stability (ORB + Homography): Dynamic ROI recalibration. Even if the camera shakes due to wind or vibrations, the parking polygons remain digitally "anchored" to the ground.
24/7 Robustness: Advanced image processing (Gamma Correction, Bloom Effect, Gaussian Noise) ensures reliability in night mode or under harsh artificial lighting.

# **Decision Logic & Stability (Finite State Machine)**
To ensure a stable database and avoid false positives (e.g., vehicles simply driving through an empty spot), we implemented a Smart State Management system:

--State	Condition	Action--

Free (0)	Occupancy < 45%	Spot available (Green)

Progress (1)	Occupancy > 45%	stability_counter starts

Occupied (2)	15 stable frames	Spot confirmed occupied (Red)

--Safety Hysteresis--

The exit threshold (to switch back from Occupied to Free) is set at 20%.
The "Why": This prevents "flickering" if the AI briefly loses a part of the vehicle due to reflections or shadows. It guarantees 100% data integrity for the dashboard and reporting.
# **Tech Stack**
AI & Vision: YOLO v11 (Ultralytics), OpenCV, ORB (Tracking & Homography).
Data Processing: Python, Shapely (Geometry), Pickle, JSON.
Web Dashboard: HTML5, CSS3, JavaScript (Fetch API for real-time monitoring).

# **Pipeline Architecture**
Load Reduction: Processes 1 out of every 3 frames (FRAME_SKIP).
ORB Tracking: Analyzes camera movement and performs image registration/alignment.
YOLO Filtering: Runs inference targeting only vehicle classes.
Intersection Calculation: Mathematical measurement of the exact occupancy rate per spot.
Validation: Passes through the State Machine before display and JSON export.

# **Installation & Usage**
Prerequisites

git clone https://github.com/yoyo2-hub/Intelligent_Parking_Monitoring_System.git

cd smart_parkingcv

pip install -r requirements.txt

# **How to use**

1-Define Spots (ROI Selection): Run the selector script to click and define the 4 corners of each parking spot.

2-Run Monitoring: Execute the main video analysis script.

3-View Dashboard: Open the web interface to track parking status live.

# **Project Team**

Chayma Dallel

Emna Ghorbel

Amir Shaier

