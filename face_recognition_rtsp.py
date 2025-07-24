import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from insightface.app import FaceAnalysis
import psycopg2

# === CONFIG ===
DB_URL = "postgresql://faceuser:gruqofbpAImi7EY6tyrGQjVsmMgMPiG6@dpg-d1oiqqadbo4c73b4fca0-a.frankfurt-postgres.render.com/face_db_7r21"
# Replace YOUR_HOST:PORT with your actual DB host from Render

# === Load known and blacklisted embeddings ===
with open("registered_faces.pkl", "rb") as f:
    registered_faces = pickle.load(f)

with open("blacklist_faces.pkl", "rb") as f:
    blacklist_faces = pickle.load(f)

# === Create folders ===
os.makedirs("unknown_faces", exist_ok=True)
os.makedirs("blacklist_images", exist_ok=True)
os.makedirs("attendance", exist_ok=True)

# === Initialize face analysis ===
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# === Save attendance ===
def mark_attendance(name):
    filepath = os.path.join("attendance", "attendance.csv")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a") as f:
        f.write(f"{name},{now}\n")

# === Save to PostgreSQL ===
def insert_unknown_face(image_path, similarity):
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO unknown_faces (image_path, similarity_score)
            VALUES (%s, %s)
        """, (image_path, similarity))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Logged unknown face to PostgreSQL")
    except Exception as e:
        print(f"❌ Database error: {e}")

# === Save unknown person ===
def save_unknown_face(face_crop):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_faces/unknown_{timestamp}.jpg"
    cv2.imwrite(filename, face_crop)
    return filename

# === Save blacklist entry ===
def save_blacklist_face(name, face_crop):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blacklist_images/{name}_{timestamp}.jpg"
    cv2.imwrite(filename, face_crop)

# === Find match by cosine similarity ===
def find_match(face_embedding, face_dict, threshold=0.4):
    best_match = "Unknown"
    best_score = -1
    for name, emb in face_dict.items():
        sim = np.dot(face_embedding, emb) / (
            np.linalg.norm(face_embedding) * np.linalg.norm(emb)
        )
        if sim > best_score and sim > threshold:
            best_score = sim
            best_match = name
    return best_match

# === Capture from webcam or RTSP ===
cap = cv2.VideoCapture(0)  # Replace 0 with RTSP URL if needed

if not cap.isOpened():
    print("[❌] Failed to open video stream.")
    exit()

print("[✅] Video stream opened. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[❌] Frame read failed.")
        break

    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Check in blacklist first
        name = find_match(embedding, blacklist_faces)
        if name != "Unknown":
            color = (0, 0, 255)  # Red
            save_blacklist_face(name, face_crop)
        else:
            # Check in registered
            name = find_match(embedding, registered_faces)
            if name == "Unknown":
                color = (0, 255, 255)  # Yellow
                image_path = save_unknown_face(face_crop)
                insert_unknown_face(image_path, 0.0)  # Adjust similarity if needed
            else:
                color = (0, 255, 0)  # Green
                mark_attendance(name)

        # Draw bounding box and label
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition RTSP/Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
