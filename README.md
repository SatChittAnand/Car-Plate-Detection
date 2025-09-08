

# 🚗 Vehicle License Plate Detection & Recognition

A modular computer vision pipeline for detecting and recognizing vehicle license plates from video footage. Built with YOLO for object detection, EasyOCR for text recognition, and SORT for real-time tracking — this system is designed for reproducibility, clarity, and performance.

---

## ✨ Key Features

- 🔍 *Vehicle Detection*  
  Detects vehicles in each frame using a pre-trained YOLOv8 model.

- 🧭 *License Plate Localization*  
  Identifies license plate regions within detected vehicles.

- 🔠 *OCR with EasyOCR*  
  Extracts alphanumeric text from license plates with confidence scoring.

- 🧠 *Vehicle Tracking (SORT)*  
  Assigns persistent IDs to vehicles across frames for consistent tracking.

- 📊 *Structured Data Export*  
  Outputs results to test.csv with:
  - Frame number  
  - Vehicle ID  
  - Bounding box coordinates  
  - Recognized license plate text  
  - OCR confidence score

---

## 🗂 Project Structure

| File | Purpose |
|------|---------|
| main.py | Orchestrates detection, tracking, OCR, and logging |
| add_missing_data.py | Fills gaps in detection data for completeness |
| sort.py | Implements the SORT tracking algorithm |
| util.py | Utility functions for OCR, formatting, validation, and CSV writing |
| visualize.py | Visualizes bounding boxes and recognized text on video frames |
| requirements.txt | Lists required Python packages |
| test.csv | Output file with detection and recognition results |
| demo.mp4, sample.mp4 | Sample input videos |
| best.pt, yolov8n.pt, license_plate_detector.pt | Pre-trained model weights |

---

## 🚀 Getting Started

### 1. Install Dependencies

Ensure you have Python ≥ 3.8 and install required packages:

bash
pip install -r requirements.txt


### 2. Run the Pipeline

To process a video and generate results:

bash
python main.py


This will analyze the input video and update test.csv with detection and recognition results.

---

## 📌 Notes

- The system is designed to be modular — you can swap models, update OCR logic, or integrate with other tracking algorithms.
- For large-scale deployment, consider batching frames and parallelizing OCR.
- If your video resolution or lighting varies significantly, retraining or fine-tuning the YOLO model may improve accuracy.

---

## 🧪 Sample Output

Here’s a snippet from test.csv:


frame_id,vehicle_id,x1,y1,x2,y2,plate_text,confidence
42,3,120,85,220,160,OD02AB1234,0.87
43,3,122,88,222,162,OD02AB1234,0.89


---

## 🤝 Contributing

Pull requests are welcome! If you’d like to improve detection accuracy, add new OCR post-processing, or enhance visualization, feel free to fork and submit changes.

---
