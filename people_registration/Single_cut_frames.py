import cv2
import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

def extract_faces(name, video_path, output_folder="known_faces", save_limit=20, skip_frames=4, similarity_threshold=0.98):
    person_folder = os.path.join(output_folder, name)
    os.makedirs(person_folder, exist_ok=True)  # Ensure folder exists

    model = YOLO("yolov8n-face.pt")

    def get_face_signature(face_crop):
        resized = cv2.resize(face_crop, (64, 64))
        return resized.flatten().astype(np.float32) / 255.0

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved = 0
    face_signatures = []

    while cap.isOpened() and saved < save_limit:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        results = model(frame)[0]
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0 or (x2 - x1) < 30 or (y2 - y1) < 30:
                continue

            signature = get_face_signature(face_crop)
            if any(cosine_similarity([signature], [sig])[0][0] > similarity_threshold for sig in face_signatures):
                continue

            face_signatures.append(signature)
            save_path = os.path.join(person_folder, f"{saved + 1}.jpeg")
            cv2.imwrite(save_path, face_crop)
            saved += 1
            print(f"Saved face {saved} for {name}")

            if saved >= save_limit:
                break

        frame_count += 1

    cap.release()
    print(f"✅ Done extracting faces for {name}. Total saved: {saved}")

if __name__ == "__main__":
    # For standalone run, replace with actual args
    extract_faces("Elie", r"C:\Users\OWNER\Desktop\WhatsApp Video 2025-05-29 at 19.06.36_073f7ecb.mp4")
