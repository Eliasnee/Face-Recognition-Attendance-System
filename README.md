# Face Recognition Attendance System

## Overview
This project is a multi-camera face recognition attendance system. It uses YOLO for face detection and a Siamese neural network for face recognition. The system tracks employee attendance, lateness, and supports adding new people via transfer learning. All attendance data is stored in CSV files for easy reporting.

## Features
- Real-time face detection and recognition
- Multi-camera (RTSP/USB) support
- Attendance and lateness tracking
- Transfer learning to add new people
- Daily and monthly attendance reports
- Modern, modular Python codebase

## Directory Structure
```
Classification/
├── README.md
├── siamese_model_finetuned.pth
├── yolov8n-cls-custom.pt
├── yolov8n-face.pt
├── __pycache__/
├── Attendance tracking/
│   ├── employee_schedule.csv
├── people registration/
│   ├── add_new_person.py
│   ├── batch_process_videos.py
│   ├── Single_cut_frames.py
│   └── __pycache__/
├── reference_faces/
│   ├── person1/
│   ├── person2/
├── siamese/
│   ├── detection_siamese.py
│   ├── multi_camera_detection.py
│   ├── train_siamese.py
│   └── __pycache__/
```

## How It Works
1. **Face Detection:** YOLO detects faces in video frames.
2. **Face Recognition:** Siamese network compares detected faces to reference images.
3. **Attendance Tracking:** Arrival times are logged and compared to the schedule.
4. **Reporting:** Daily and monthly CSVs are generated in `Attendance tracking/`.
5. **Adding New People:** Use `add_new_person.py` or `batch_process_videos.py` to add new faces and retrain the model.

## Usage
### 1. Install Requirements
```sh
pip install torch torchvision ultralytics opencv-python pandas pillow facenet-pytorch
```

### 2. Prepare Data
- Place reference images in `reference_faces/PersonName/`
- Edit `Attendance tracking/employee_schedule.csv` for employee names and expected arrival times

### 3. Train or Fine-tune Model
- Initial training: `python siamese/train_siamese.py`
- Add new person: `python siamese/add_new_person.py "Name" --model models/best_siamese_model.pth --images reference_faces/Name`

### 4. Run Detection
- Single camera: `python siamese/detection_siamese.py`
- Multi-camera: `python siamese/multi_camera_detection.py`

#### Changing the Camera Source
- **For a USB webcam:**
  - In `detection_siamese.py`, change the line:
    ```python
    cap = cv2.VideoCapture(0)
    ```
    Use `0` for the default webcam, `1` for a second webcam, etc.
- **For an RTSP (IP) camera:**
  - In `detection_siamese.py`, change the line to:
    ```python
    cap = cv2.VideoCapture('rtsp://username:password@ip_address:port/stream')
    ```
    Replace with your camera's RTSP URL.
- **For multi-camera setup:**
  - In `multi_camera_detection.py`, edit the `cameras` dictionary:
    ```python
    cameras = {
        "Main_Entrance": {"url": "rtsp://admin:password@192.168.1.100:554/stream1"},
        "Side_Door": {"url": 0}  # 0 for webcam, or use another RTSP URL
    }
    ```
  - You can mix RTSP streams and webcam indices as needed.

### 5. View Attendance
- Check `Attendance tracking/attendance_YYYY-MM-DD.csv` and `monthly_attendance_YYYY-MM.csv`

## Adding New People
- Use `add_new_person.py` for one person, or `batch_process_videos.py` for many.
- The system will fine-tune the model and update detection scripts automatically.

## Attendance CSVs
- **employee_schedule.csv:** Name, Expected_Arrival
- **attendance_YYYY-MM-DD.csv:** Name, Arrival_Time, Minutes_Late, Camera_Location
- **monthly_attendance_YYYY-MM.csv:** Date, Name, Minutes_Late, Status, Camera_Location

## How to Store Reference Faces
- Create a folder named `reference_faces` in your project root if it does not exist.
- For each person, create a subfolder inside `reference_faces` with the person's name (e.g., `reference_faces/John/`).
- Place 10–20 clear, front-facing images of the person's face in their folder. Use `.jpg` or `.jpeg` format.
- Example structure:

```
reference_faces/
├── John/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── Emma/
│   ├── 1.jpg
│   └── ...
└── Unknown/
    ├── 1.jpg
    └── ...
```
- The `Unknown` folder should contain faces of people not in your database, to help the model learn to reject unknowns.
- Make sure each image contains only one face and is not blurry or occluded.

## Tips
- Use a CUDA GPU for best performance
- Use RTSP URLs for IP cameras, or 0/1 for USB webcams
- Keep `reference_faces/Unknown/` populated for better unknown detection

## Authors
- Your Team Name / Contributors

## License
MIT License
