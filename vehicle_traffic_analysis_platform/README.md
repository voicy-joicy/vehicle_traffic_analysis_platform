# Vehicle Traffic Analysis Platform

This is a working Streamlit application for detecting vehicles in uploaded traffic videos using YOLOv8, OpenCV, and Python.

## Main Features

- Upload a traffic video.
- Process the video frame by frame.
- Detect vehicle classes using YOLOv8.
- Track vehicles across frames using a ByteTrack-style tracker.
- Draw bounding boxes, stable track IDs, and class labels.
- Count vehicle detections and tracked vehicles.
- Generate and download an annotated output video.
- Display detection, tracking, and direction analytics in a dashboard.

## Project Structure

```text
vehicle_traffic_analysis_platform/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── .gitkeep
│
├── uploads/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── sample_videos/
│   └── .gitkeep
│
└── utils/
    └── detection_utils.py
```

## Installation

### 1. Open the project folder

```bash
cd vehicle_traffic_analysis_platform
```

### 2. Create a virtual environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

The app will open in your browser. Upload a traffic video and click **Start Vehicle Detection**.

## Notes

- The first run may take longer because YOLOv8 may download the selected model file automatically.
- Use `yolov8n.pt` for faster processing.
- Use `yolov8s.pt` or `yolov8m.pt` for potentially better accuracy, but processing will be slower.
- This version counts vehicle detections, not exact unique vehicles. Add tracking and line-crossing logic for exact vehicle counting.

## Vehicle Classes Used

The pretrained YOLOv8 COCO classes used in this project are:

- bicycle
- car
- motorcycle
- bus
- truck
