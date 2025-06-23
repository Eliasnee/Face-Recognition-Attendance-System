import cv2
import threading
from pathlib import Path
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from train_siamese import SiameseNetwork
from torchvision import transforms
from PIL import Image
import queue
import time
import pandas as pd
from datetime import datetime
import numpy as np
import os
from multiprocessing import cpu_count
import concurrent.futures
from collections import deque
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Camera stream class to handle RTSP stream reading in separate threads
class CameraStream:
    def __init__(self, camera_url, camera_id):
        self.camera_url = camera_url
        self.camera_id = camera_id
        self.frame_queue = queue.Queue(maxsize=30)
        self.stopped = False

    def start(self):
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()
        return self

    def update(self):
        while not self.stopped:
            cap = cv2.VideoCapture(self.camera_url)
            if not cap.isOpened():
                logger.warning(f"Camera {self.camera_id}: Unable to open stream, retrying in 5s...")
                time.sleep(5)
                continue

            while not self.stopped:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Camera {self.camera_id}: Frame not received, reconnecting...")
                    break  # reconnect stream
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
                time.sleep(0.01)

            cap.release()

    def read(self):
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True

class FaceDetectionSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.yolo_face = YOLO("yolov8n-face.pt").to(self.device)
        self.siamese_net = SiameseNetwork()
        self.siamese_net.load_state_dict(torch.load("siamese_model_finetuned.pth", map_location=self.device))
        self.siamese_net.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.SIMILARITY_THRESHOLD = 0.65
        self.MIN_REFERENCES_MATCH = 3
        self.REFERENCE_DIR = "reference_faces"

        self.reference_embeddings = self.load_reference_faces()

        self.frame_history_size = 15
        self.min_consistent_frames = 10

    def load_reference_faces(self):
        reference_embeddings = {}
        reference_dir = Path(self.REFERENCE_DIR)

        for person_dir in reference_dir.glob('*/'):
            if person_dir.name.lower() == 'unknown':
                continue
            embeddings = []
            for img_path in person_dir.glob('*.jpg'):
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.siamese_net.forward_one(img_tensor)
                embeddings.append(embedding)
            if embeddings:
                reference_embeddings[person_dir.name] = embeddings
        return reference_embeddings

    def find_best_match(self, face_embedding, detected_faces_count):
        matches, similarities = {}, {}
        threshold = self.SIMILARITY_THRESHOLD - 0.02 if detected_faces_count > 1 else self.SIMILARITY_THRESHOLD

        for name, refs in self.reference_embeddings.items():
            sim_list = [F.cosine_similarity(face_embedding, r).item() for r in refs]
            match_count = sum(s > threshold for s in sim_list)

            if match_count >= self.MIN_REFERENCES_MATCH:
                avg_sim = sum(sorted(sim_list, reverse=True)[:self.MIN_REFERENCES_MATCH]) / self.MIN_REFERENCES_MATCH
                matches[name] = match_count
                similarities[name] = avg_sim

        if not matches:
            return "Unknown", 0.0

        best = max(matches.items(), key=lambda x: (x[1], similarities[x[0]]))
        if best[1] >= self.MIN_REFERENCES_MATCH:
            return best[0], similarities[best[0]]
        else:
            return "Unknown", 0.0

    def record_attendance(self, identity, camera_id, similarity):
        if identity == "Unknown" or similarity < self.SIMILARITY_THRESHOLD:
            return

        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        os.makedirs('Attendance tracking', exist_ok=True)
        schedule_path = 'Attendance tracking/employee_schedule.csv'

        if not os.path.exists(schedule_path):
            logger.warning("Schedule file not found.")
            return

        schedule_df = pd.read_csv(schedule_path)
        if identity not in schedule_df['Name'].values:
            return

        expected_str = schedule_df[schedule_df['Name'] == identity]['Expected_Arrival'].iloc[0]
        expected_time = datetime.strptime(f"{today} {expected_str}", '%Y-%m-%d %H:%M')
        minutes_late = max(0, (now - expected_time).total_seconds() / 60)

        daily_path = f'Attendance tracking/attendance_{today}.csv'
        daily_df = pd.read_csv(daily_path) if os.path.exists(daily_path) else pd.DataFrame(columns=['Name', 'Arrival_Time', 'Minutes_Late', 'Camera_Location'])
        if identity in daily_df['Name'].values:
            return

        new_daily = pd.DataFrame([{
            'Name': identity,
            'Arrival_Time': now.strftime('%H:%M:%S'),
            'Minutes_Late': round(minutes_late),
            'Camera_Location': camera_id
        }])
        pd.concat([daily_df, new_daily], ignore_index=True).to_csv(daily_path, index=False)

        month_file = f'Attendance tracking/monthly_attendance_{now.strftime("%Y-%m")}.csv'
        monthly_df = pd.read_csv(month_file) if os.path.exists(month_file) else pd.DataFrame(columns=['Date', 'Name', 'Minutes_Late', 'Status', 'Camera_Location'])
        status = 'On Time'
        if minutes_late > 60:
            status = 'Very Late'
        elif minutes_late > 30:
            status = 'Late'
        elif minutes_late > 5:
            status = 'Slightly Late'

        new_monthly = pd.DataFrame([{
            'Date': today,
            'Name': identity,
            'Minutes_Late': round(minutes_late),
            'Status': status,
            'Camera_Location': camera_id
        }])
        pd.concat([monthly_df, new_monthly], ignore_index=True).to_csv(month_file, index=False)

    def process_camera(self, stream: CameraStream, camera_id: str):
        performance_monitor = PerformanceMonitor()
        frame_history: List[Tuple[str, float]] = []

        while not stream.stopped:
            frame = stream.read()
            if frame is None:
                continue

            frame_start_time = time.time()

            results = self.yolo_face(frame)[0]
            detected_faces_count = len(results.boxes)
            identities_in_frame = []

            faces = []
            boxes = []

            h, w = frame.shape[:2]

            # Extract faces and boxes first
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Add margin
                margin = 0.2
                x1 = max(0, int(x1 - (x2 - x1) * margin))
                y1 = max(0, int(y1 - (y2 - y1) * margin))
                x2 = min(w, int(x2 + (x2 - x1) * margin))
                y2 = min(h, int(y2 + (y2 - y1) * margin))

                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                faces.append(face_img)
                boxes.append((x1, y1, x2, y2))

            if len(faces) == 0:
                cv2.putText(frame, f"Camera: {camera_id} - No faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(f"Camera {camera_id}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Process all faces embeddings at once for speed
            face_tensors = []
            for face_img in faces:
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensors.append(self.transform(face_pil).unsqueeze(0))
            face_tensors = torch.cat(face_tensors).to(self.device)

            with torch.no_grad():
                embeddings = self.siamese_net.forward_one(face_tensors)

            # Match embeddings to reference faces
            current_frame_identities = []
            current_frame_sims = []

            for emb in embeddings:
                identity, sim = self.find_best_match(emb.unsqueeze(0), detected_faces_count)
                current_frame_identities.append(identity)
                current_frame_sims.append(sim)

            # Add frame history and temporal filtering
            frame_history.append(list(zip(current_frame_identities, current_frame_sims)))
            if len(frame_history) > self.frame_history_size:
                frame_history.pop(0)

            reliable_identities = []
            for i in range(len(current_frame_identities)):
                counts = {}
                sims = {}
                for hist_frame in frame_history:
                    id, sim = hist_frame[i]
                    counts[id] = counts.get(id, 0) + 1
                    sims.setdefault(id, []).append(sim)

                most_common = None
                highest_count = 0

                required_frames = self.min_consistent_frames - 2 if detected_faces_count > 1 else self.min_consistent_frames
                required_similarity = self.SIMILARITY_THRESHOLD - 0.02 if detected_faces_count > 1 else self.SIMILARITY_THRESHOLD

                for id, count in counts.items():
                    avg_sim = sum(sims[id]) / len(sims[id])
                    if count >= required_frames and avg_sim > required_similarity:
                        if count > highest_count:
                            most_common = (id, count)
                            highest_count = count

                if most_common is not None:
                    reliable_identities.append(most_common[0])
                    self.record_attendance(most_common[0], camera_id, required_similarity)
                else:
                    reliable_identities.append("Unknown")

            # Draw boxes and labels
            for (x1, y1, x2, y2), identity, similarity in zip(boxes, reliable_identities, current_frame_sims):
                if similarity > self.SIMILARITY_THRESHOLD:
                    color = (0, 255, 0) if similarity > 0.92 else (0, 255, 255)
                    label = f"{identity} ({similarity:.3f})"
                    conf_text = "High Confidence" if similarity > 0.92 else "Low Confidence"
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({similarity:.3f})"
                    conf_text = "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f"Camera: {camera_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(f"Camera {camera_id}", frame)

            frame_processing_time = time.time() - frame_start_time
            performance_monitor.update_camera_stats(camera_id, frame_processing_time, detected_faces_count)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(f"Camera {camera_id}")

    def run(self, camera_urls):
        streams = []
        threads = []

        for i, url in enumerate(camera_urls, 1):
            stream = CameraStream(url, f"Camera {i}").start()
            streams.append(stream)

        for stream in streams:
            t = threading.Thread(target=self.process_camera, args=(stream, stream.camera_id), daemon=True)
            t.start()
            threads.append(t)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping all streams and threads...")
            for stream in streams:
                stream.stop()
            for t in threads:
                t.join()
            cv2.destroyAllWindows()

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = {}
        self.face_counts = {}

    def update_camera_stats(self, camera_id, frame_time, face_count):
        if camera_id not in self.frame_times:
            self.frame_times[camera_id] = deque(maxlen=100)
            self.face_counts[camera_id] = deque(maxlen=100)

        self.frame_times[camera_id].append(frame_time)
        self.face_counts[camera_id].append(face_count)

        avg_time = sum(self.frame_times[camera_id]) / len(self.frame_times[camera_id])
        avg_faces = sum(self.face_counts[camera_id]) / len(self.face_counts[camera_id])
        logger.info(f"[{camera_id}] Avg frame time: {avg_time:.3f}s, Avg faces: {avg_faces:.2f}")

if __name__ == "__main__":
    face_system = FaceDetectionSystem()
    # Add your RTSP stream URLs here
    camera_streams = [
        0,
        r"C:\Users\OWNER\Desktop\WhatsApp Video 2025-06-23 at 09.34.06_dc6cd510.mp4",
        # Add more streams if you want
    ]
    face_system.run(camera_streams)
