from ultralytics import YOLO
import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
import dlib
from train_siamese import SiameseNetwork
import pandas as pd
from datetime import datetime, timedelta
import os

# Load models
device = torch.device('cpu')  # Force CPU usage
yolo_face = YOLO(r"yolov8n-face.pt")  # Face detector
yolo_face.to(device)  # Ensure YOLO runs on CPU

# Load Siamese network
siamese_net = SiameseNetwork()
siamese_net.load_state_dict(torch.load(r"siamese_model_finetuned.pth", map_location=device))
siamese_net.eval()
siamese_net = siamese_net.to(device)

# Constants
SIMILARITY_THRESHOLD = 0.65  # Reduced base threshold
REFERENCE_DIR = "reference_faces"
MIN_REFERENCES_MATCH = 3  # Reduced reference matches required
FRAME_HISTORY_SIZE = 15  # Increased frame history for better temporal consistency
MIN_CONSISTENT_FRAMES = 10  # Reduced consistent frames required

# Initialize frame history
frame_history = []
last_reliable_identity = "Unknown"

# Setup transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load reference faces
reference_embeddings = {}
reference_dir = Path(REFERENCE_DIR)

def load_reference_faces():
    # Create person folders if they don't exist
    for person_img in reference_dir.glob('*.jpg'):
        person_name = person_img.stem
        person_dir = reference_dir / person_name
        person_dir.mkdir(exist_ok=True)
        if not list(person_dir.glob('*.jpg')):  # If folder is empty
            shutil.copy2(person_img, person_dir / f"{person_name}_ref.jpg")
    
    # Load embeddings from person folders
    reference_embeddings.clear()
    for person_dir in reference_dir.glob('*/'):
        person_name = person_dir.name
        embeddings = []
        
        for img_path in person_dir.glob('*.jpg'):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = siamese_net.forward_one(img_tensor)
            embeddings.append(embedding)
        
        if embeddings:
            reference_embeddings[person_name] = embeddings

def find_best_match(face_embedding, detected_faces_count):
    matches = {}
    similarities = {}
      # Adjust matching criteria for multiple faces - keep thresholds consistent
    if detected_faces_count > 1:
        local_similarity_threshold = SIMILARITY_THRESHOLD - 0.02  # Slightly more lenient for multiple faces
        local_min_references = MIN_REFERENCES_MATCH
    else:
        local_similarity_threshold = SIMILARITY_THRESHOLD
        local_min_references = MIN_REFERENCES_MATCH
    
    # Calculate similarities with all reference images for each person
    for person_name, ref_embeddings_list in reference_embeddings.items():
        person_similarities = []
        match_count = 0
        top_similarities = []
          # Get initial similarity scores
        for ref_embedding in ref_embeddings_list:
            similarity = F.cosine_similarity(face_embedding, ref_embedding).item()
            person_similarities.append(similarity)
            if similarity > local_similarity_threshold:  # Use local threshold
                match_count += 1
                top_similarities.append(similarity)
                  # Early success - adjusted for multiple faces
                if match_count >= local_min_references:
                    # Additional verification - check if average of top matches is good
                    top_n = sorted(person_similarities, reverse=True)[:local_min_references]
                    avg_similarity = sum(top_n) / len(top_n)
                    
                    # Use different thresholds based on number of faces
                    if detected_faces_count > 1:
                        if avg_similarity > 0.80:  # More lenient for multiple faces
                            matches[person_name] = match_count
                            similarities[person_name] = avg_similarity
                            return person_name, avg_similarity
                    else:
                        if avg_similarity > 0.85:  # Original threshold for single face
                            matches[person_name] = match_count
                            similarities[person_name] = avg_similarity
                            return person_name, avg_similarity
        
        # Sort similarities and take top 3
        top_similarities = sorted(person_similarities, reverse=True)[:3]
        
        if len(top_similarities) > 0:
            avg_top_similarity = sum(top_similarities) / len(top_similarities)
            matches[person_name] = match_count
            similarities[person_name] = avg_top_similarity
    
    # Find the person with the most matches above threshold
    if not matches:
        return "Unknown", 0.0
        
    best_person = max(matches.items(), key=lambda x: (x[1], similarities[x[0]]))
      # Only return a match if we have enough reference matches
    if best_person[1] >= local_min_references:  # Use local reference count
        return best_person[0], similarities[best_person[0]]
    
    return "Unknown", 0.0

# Load reference faces
load_reference_faces()

# Attendance tracking functions
def load_employee_schedule():
    schedule_path = 'Attendance tracking/employee_schedule.csv'
    os.makedirs('Attendance tracking', exist_ok=True)
    
    if os.path.exists(schedule_path):
        return pd.read_csv(schedule_path)
    
    # Create default schedule if it doesn't exist
    default_schedule = pd.DataFrame({
        'Name': ['Elias', 'Elio', 'Tia'],
        'Expected_Arrival': ['09:00', '09:00', '09:00']
    })
    default_schedule.to_csv(schedule_path, index=False)
    return default_schedule

def get_daily_attendance_file():
    today = datetime.now().strftime('%Y-%m-%d')
    file_path = f'Attendance tracking/attendance_{today}.csv'
    os.makedirs('Attendance tracking', exist_ok=True)
    
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=['Name', 'Arrival_Time', 'Minutes_Late', 'Camera_Location'])
        df.to_csv(file_path, index=False)
    return file_path

def get_monthly_attendance_file():
    month_year = datetime.now().strftime('%Y-%m')
    file_path = f'Attendance tracking/monthly_attendance_{month_year}.csv'
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=['Date', 'Name', 'Minutes_Late', 'Status'])
        df.to_csv(file_path, index=False)
    return file_path

def record_attendance(identity, similarity):
    if identity == "Unknown" or similarity < SIMILARITY_THRESHOLD:
        return

    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    
    # Load employee schedule
    schedule_df = load_employee_schedule()
    if identity not in schedule_df['Name'].values:
        return
    
    # Get expected arrival time
    expected_time_str = schedule_df[schedule_df['Name'] == identity]['Expected_Arrival'].iloc[0]
    expected_time = datetime.strptime(f"{today} {expected_time_str}", '%Y-%m-%d %H:%M')
    
    # Get daily attendance file
    daily_file = get_daily_attendance_file()
    daily_df = pd.read_csv(daily_file)
    
    # Check if already recorded for today
    if identity in daily_df['Name'].values:
        return
    
    # Calculate lateness
    minutes_late = max(0, (now - expected_time).total_seconds() / 60)
    
    # Record daily attendance
    new_record = pd.DataFrame({
        'Name': [identity],
        'Arrival_Time': [now.strftime('%H:%M:%S')],
        'Minutes_Late': [round(minutes_late)]
    })
    daily_df = pd.concat([daily_df, new_record], ignore_index=True)
    daily_df.to_csv(daily_file, index=False)
    
    # Update monthly attendance
    monthly_file = get_monthly_attendance_file()
    monthly_df = pd.read_csv(monthly_file)
    
    # Determine status
    status = 'On Time'
    if minutes_late > 60:
        status = 'Very Late'
    elif minutes_late > 30:
        status = 'Late'
    elif minutes_late > 5:
        status = 'Slightly Late'
    
    new_monthly_record = pd.DataFrame({
        'Date': [today],
        'Name': [identity],
        'Minutes_Late': [round(minutes_late)],
        'Status': [status]
    })
    monthly_df = pd.concat([monthly_df, new_monthly_record], ignore_index=True)
    monthly_df.to_csv(monthly_file, index=False)

# Initialize attendance tracking
employee_schedule = load_employee_schedule()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break    # Detect faces
    results = yolo_face(frame)[0]
    detected_faces_count = len(results.boxes)  # Count detected faces
    def add_margin(x1, y1, x2, y2, margin=0.2):
        width = x2 - x1
        height = y2 - y1
        x1 = max(0, int(x1 - width * margin))
        y1 = max(0, int(y1 - height * margin))
        x2 = min(frame.shape[1], int(x2 + width * margin))
        y2 = min(frame.shape[0], int(y2 + height * margin))
        return x1, y1, x2, y2

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Add margin to include more context
        x1, y1, x2, y2 = add_margin(x1, y1, x2, y2)
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        # Convert to LAB color space for better lighting handling
        lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with more aggressive parameters
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        l = clahe.apply(l)
        
        # Merge back and convert to BGR
        lab = cv2.merge([l, a, b])
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Additional preprocessing
        # 1. Gamma correction
        gamma = 1.2
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        normalized = cv2.LUT(normalized, lookUpTable)
        
        # 2. Increase contrast
        alpha = 1.3  # Contrast control
        beta = 0     # Brightness control
        normalized = cv2.convertScaleAbs(normalized, alpha=alpha, beta=beta)
          # Convert to PIL and get embedding
        cropped_pil = Image.fromarray(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
        face_tensor = transform(cropped_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            face_embedding = siamese_net.forward_one(face_tensor)
          # Find best match
        identity, similarity = find_best_match(face_embedding, detected_faces_count)
          # Apply temporal consistency check with similarity scores
        frame_history.append((identity, similarity))
        if len(frame_history) > FRAME_HISTORY_SIZE:
            frame_history.pop(0)
            
        # Only update identity if we have enough frames
        if len(frame_history) == FRAME_HISTORY_SIZE:
            # Count occurrences of each identity and track average similarity
            identity_counts = {}
            identity_similarities = {}
            for id, sim in frame_history:
                identity_counts[id] = identity_counts.get(id, 0) + 1
                if id not in identity_similarities:
                    identity_similarities[id] = []
                identity_similarities[id].append(sim)            # Find most common identity with high average similarity
            most_common = None
            highest_count = 0
            required_frames = MIN_CONSISTENT_FRAMES
            required_similarity = SIMILARITY_THRESHOLD
              # Adjust consistency requirements for multiple faces - more lenient
            if detected_faces_count > 1:
                required_frames = MIN_CONSISTENT_FRAMES - 2  # Require fewer consistent frames
                required_similarity = SIMILARITY_THRESHOLD - 0.02  # Lower similarity threshold
            
            for id, count in identity_counts.items():
                avg_similarity = sum(identity_similarities[id]) / len(identity_similarities[id])
                if count >= required_frames and avg_similarity > required_similarity:
                    if count > highest_count:
                        most_common = (id, count)
                        highest_count = count
              # Only accept identity if it appears in enough frames with high confidence
            if most_common is not None:
                identity = most_common[0]
                last_reliable_identity = identity
                # Record attendance when identity is confirmed
                record_attendance(identity, similarity)
            else:
                # If no consistent identity, mark as Unknown
                identity = "Unknown"
        
        # Draw results with enhanced visualization
        if similarity > SIMILARITY_THRESHOLD:
            if similarity > 0.92:  # Very confident match
                color = (0, 255, 0)  # Green
                conf_text = "High Confidence"
            else:  # Less confident match
                color = (0, 255, 255)  # Yellow
                conf_text = "Low Confidence"
            label = f"{identity} ({similarity:.3f})"
        else:
            color = (0, 0, 255)  # Red
            conf_text = "Unknown"
            label = f"Unknown ({similarity:.3f})"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw labels with confidence
        cv2.putText(frame, label, (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, conf_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the annotated frame
    cv2.imshow("Siamese Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()