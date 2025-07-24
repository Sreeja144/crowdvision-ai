import os
import cv2
import pickle
from insightface.app import FaceAnalysis

faces_dir = "faces"
blacklist_dir = "blacklist"

registered_faces = {}
blacklist_faces = {}

# Initialize FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[❌] Cannot read image: {image_path}")
        return None
    faces = app.get(img)
    if len(faces) == 0:
        print(f"[⚠️] No face detected in: {image_path}")
        return None
    return faces[0].embedding

# Process blacklist
print("🚫 Blacklisted Faces Detected:")
for file in os.listdir(blacklist_dir):
    name = os.path.splitext(file)[0].strip().lower()
    path = os.path.join(blacklist_dir, file)
    emb = get_embedding(path)
    if emb is not None:
        blacklist_faces[name] = emb
        print(f"   - {name}")

# Process known faces (excluding blacklist names)
print("\n✅ Registered Faces:")
for file in os.listdir(faces_dir):
    name = os.path.splitext(file)[0].strip().lower()
    if name in blacklist_faces:
        print(f"   - {name} [⚠️ Skipped, in blacklist]")
        continue
    path = os.path.join(faces_dir, file)
    emb = get_embedding(path)
    if emb is not None:
        registered_faces[name] = emb
        print(f"   - {name}")

# Save both sets
with open("registered_faces.pkl", "wb") as f:
    pickle.dump(registered_faces, f)

with open("blacklist_faces.pkl", "wb") as f:
    pickle.dump(blacklist_faces, f)

print("\n✅ Registration Complete")
print(f"   ✅ Total Registered: {len(registered_faces)}")
print(f"   🚫 Total Blacklisted: {len(blacklist_faces)}")
