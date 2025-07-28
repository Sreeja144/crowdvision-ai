import os
import cv2
import numpy as np
import pickle
import threading
import time
import smtplib
import psycopg2
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from waitress import serve
import mediapipe as mp
import tempfile 
from werkzeug.utils import secure_filename

# Import Required Modules for MediaPipe Tasks API
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmarkerResult

# IMPORTANT NEW IMPORTS: Import NormalizedLandmark and NormalizedLandmarkList from the protobufs
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark 

from insightface.app import FaceAnalysis
# NEW: Import YOLO from ultralytics
from ultralytics import YOLO 
import pygame

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# MediaPipe setup (for drawing utilities)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize InsightFace
print("Initializing InsightFace model...")
try:
    face_analysis = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_analysis.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace model loaded successfully.")
except Exception as e:
    print(f"Error initializing InsightFace: {e}")
    face_analysis = None

# Initialize Pygame Mixer
try:
    pygame.mixer.init()
    print("Pygame mixer initialized.")
except Exception as e:
    print(f"Error initializing Pygame mixer: {e}")
    pygame_mixer_initialized = False
else:
    pygame_mixer_initialized = True

# Database Configuration
DATABASE_URL = "postgresql://faceuser:gruqofbpAImi7EY6tyrGQjVsmMgMPiG6@dpg-d1oiqqadbo4c73b4fca0-a.frankfurt-postgres.render.com/face_db_7r21"

class Config:
    EMAIL_SENDER = "smadala4@gitam.in"
    EMAIL_PASSWORD = "kljn nztp qqot juwe" # This should be an App Password for security (use app password)
    AUDIO_FOLDER = "audio"
    VIDEO_FOLDER = "video"
    DEFAULT_VIDEO = "vandalism.mp4" # Or standing.webm, depending on your default
    STANDING_THRESHOLD = 160
    BENDING_THRESHOLD = 150
    
    # FACE_MATCH_THRESHOLD: This value represents the maximum Euclidean distance allowed
    # between two face embeddings for them to be considered a match.
    # A LOWER value means a STRICTER match (embeddings must be more similar).
    # A HIGHER value means a MORE LENIENT match (embeddings can be less similar).
    # Common values for InsightFace with Euclidean distance are often between 0.6 and 0.8.
    FACE_MATCH_THRESHOLD = 0.65 # A more balanced value for accurate known face detection.
    
    CROWD_THRESHOLD = 5 
    
    # Lowering pose detection confidence significantly
    POSE_DETECTION_CONFIDENCE = 0.05 # Very aggressive
    
    # Corrected RESTRICTED_ZONE to have a proper width/height
    RESTRICTED_ZONE = (0, 0, 0, 0) # (x1, y1, x2, y2)
    FRAME_SKIP = 20
    ALLOWED_EXTENSIONS = {'mp4', 'webm', 'avi', 'mov'}

    # YOLO Configuration - NOW USING ULTRALYTICS YOLO WITH .PT MODEL
    YOLO_MODEL_DIR = os.path.join("models", "yolo")
    # IMPORTANT: Change this to the .pt file you download
    YOLO_MODEL_PATH = os.path.join(YOLO_MODEL_DIR, "yolov3-tiny.pt") # Changed to .pt
    # YOLO_NAMES is not strictly needed for ultralytics predict, but kept for consistency if needed elsewhere
    YOLO_NAMES = os.path.join(YOLO_MODEL_DIR, "coco.names") 
    YOLO_CONF_THRESHOLD = 0.5 # Confidence threshold for object detection
    YOLO_NMS_THRESHOLD = 0.4 # Non-maximum suppression threshold

    # NEW: Threshold for theft proximity (normalized coordinates)
    THEFT_PROXIMITY_THRESHOLD = 0.08 # Adjust this value based on testing

    # NEW: Number of poses to detect per person crop (set to 1 for single pose per person)
    NUM_POSES_PER_PERSON_CROP = 1

    # NEW: Folder for storing unknown face images
    UNKNOWN_FACES_FOLDER = "unknown_faces"

    # NEW: Tracking configuration
    TRACKING_IOU_THRESHOLD = 0.3 # IoU threshold for matching existing tracks
    TRACKING_TIMEOUT_FRAMES = 10 # Number of frames after which a person is considered gone


# Global monitoring data
monitoring_data = {
    "status": "ready", 
    "maximum_crowd_count": 0,
    "current_crowd_count": 0,
    "current_standing_count": 0, # NEW: Per-frame standing count
    "current_bending_count": 0,  # NEW: Per-frame bending count
    "standing_detections": 0,    # Total accumulated standing detections (from unique tracked persons)
    "bending_detections": 0,     # Total accumulated bending detections (from unique tracked persons)
    "unknown_faces": 0,          # Total unique unknown faces detected
    "known_faces": 0,            # Total unique known faces detected
    "theft_detected": False, 
    "restricted_zone_breached": False, 
    "last_updated": None,
    "active_clients": 0, 
    "last_frame": None, 
    "frame_counter": 0,
    "processing_frame_index": 0,
    "email_for_report": None, 
    "video_processing_complete": False,
    "current_temp_video_path": None
}

# New global variables for alert timing
last_theft_alert_time = 0
last_crowd_alert_time = 0

# Global tracking variables
# tracked_persons: {person_id: {'bbox': (x1,y1,x2,y2), 'last_seen_frame': frame_idx, 'is_unknown_face': bool, 'is_standing': bool, 'is_bending': bool, 'is_known_face': bool}}
tracked_persons = {}
next_person_id = 0


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Database functions
def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def init_db():
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            # Create events table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(50) NOT NULL,
                description TEXT,
                timestamp TIMESTAMP NOT NULL
            )""")
            # Create unknown_faces table with the new schema
            cur.execute("""
            CREATE TABLE IF NOT EXISTS unknown_faces (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT NOT NULL,
                similarity_score REAL
            )""")
            conn.commit()
            cur.close()
            print("Database initialized successfully.")
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        if conn:
            conn.close()

def log_event(event_type, description):
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO events (event_type, description, timestamp) VALUES (%s, %s, %s)",
                (event_type, description, datetime.now())
            )
            conn.commit()
            cur.close()
    except Exception as e:
        print(f"Database logging error: {e}")
    finally:
        if conn:
            conn.close()

def log_unknown_face(face_image_np, similarity_score=-1.0):
    """
    Logs an unknown face by saving its image to a file and storing the path and similarity score in the database.
    
    Args:
        face_image_np (numpy.ndarray): The cropped unknown face image (NumPy array).
        similarity_score (float): The similarity score, typically -1.0 for unknown faces.
    """
    conn = None
    try:
        # Ensure the directory for unknown faces exists
        if not os.path.exists(Config.UNKNOWN_FACES_FOLDER):
            os.makedirs(Config.UNKNOWN_FACES_FOLDER)

        # Generate a unique filename for the image using timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] # Include milliseconds
        image_filename = f"unknown_{timestamp_str}.jpg"
        image_full_path = os.path.join(Config.UNKNOWN_FACES_FOLDER, image_filename)

        # Save the image to the file system
        cv2.imwrite(image_full_path, face_image_np)
        print(f"Saved unknown face image to: {image_full_path}")

        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO unknown_faces (timestamp, image_path, similarity_score) VALUES (%s, %s, %s)",
                (datetime.now(), image_full_path, similarity_score)
            )
            conn.commit()
            cur.close()
            print(f"Logged unknown face path to database: {image_full_path}, Similarity: {similarity_score}")
    except Exception as e:
        print(f"Database logging error for unknown face: {e}")
    finally:
        if conn:
            conn.close()

def get_unknown_face_image_paths_from_db():
    """
    Retrieves all image paths from the unknown_faces table.
    """
    conn = None
    image_paths = []
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT image_path FROM unknown_faces ORDER BY timestamp ASC")
            rows = cur.fetchall()
            image_paths = [row[0] for row in rows]
            cur.close()
    except Exception as e:
        print(f"Error fetching unknown face image paths from database: {e}")
    finally:
        if conn:
            conn.close()
    return image_paths

def clear_unknown_faces_db():
    """
    Clears all entries from the unknown_faces table.
    """
    conn = None
    try:
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM unknown_faces")
            conn.commit()
            cur.close()
            print("Cleared unknown_faces table.")
    except Exception as e:
        print(f"Error clearing unknown_faces table: {e}")
    finally:
        if conn:
            conn.close()

# Face encodings loading
registered_faces = {}
blacklist_faces = {}
try:
    if os.path.exists("registered_faces.pkl"):
        with open("registered_faces.pkl", "rb") as f:
            registered_faces = pickle.load(f)
        print(f"Loaded {len(registered_faces)} registered faces.")
    if os.path.exists("blacklist_faces.pkl"):
        with open("blacklist_faces.pkl", "rb") as f:
            blacklist_faces = pickle.load(f)
        print(f"Loaded {len(blacklist_faces)} blacklisted faces.")
except Exception as e:
    print(f"Error loading face encodings: {e}")

# Face matching function
def find_match(embedding, database, threshold=Config.FACE_MATCH_THRESHOLD):
    best_name, min_dist = None, float('inf')
    for name, db_emb in database.items():
        dist = np.linalg.norm(db_emb - embedding)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            best_name = name
    return best_name

# Posture detection functions
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
    return np.degrees(np.arccos(cosine_angle))

def get_landmarks_from_object(person_landmarks_object):
    """
    Helper function to safely extract the list of landmarks,
    handling both NormalizedLandmarkList and direct list cases.
    """
    if hasattr(person_landmarks_object, 'landmark') and isinstance(person_landmarks_object.landmark, list):
        return person_landmarks_object.landmark
    elif isinstance(person_landmarks_object, list):
        return person_landmarks_object
    else:
        print(f"WARNING: Unexpected landmark object type: {type(person_landmarks_object)}. Returning None.")
        return None

def detect_standing(person_landmarks_object):
    landmarks = get_landmarks_from_object(person_landmarks_object)
    if landmarks is None:
        return False
    
    try:
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        angle = calculate_angle(left_shoulder, left_hip, left_knee) 
        return angle > Config.STANDING_THRESHOLD
    except (IndexError, AttributeError) as e:
        # print(f"Error in detect_standing: {e}. Landmarks object type inside function: {type(person_landmarks_object)}")
        return False

def detect_bending(person_landmarks_object):
    landmarks = get_landmarks_from_object(person_landmarks_object)
    if landmarks is None:
        return False
    
    try:
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        angle = calculate_angle(left_shoulder, left_hip, left_knee) 
        return angle < Config.BENDING_THRESHOLD
    except (IndexError, AttributeError) as e:
        # print(f"Error in detect_bending: {e}. Landmarks object type inside function: {type(person_landmarks_object)}")
        return False

def detect_theft(person_landmarks_object):
    # Theft detection logic is commented out as requested.
    # This function will always return False, effectively disabling theft detection.
    return False
    """
    landmarks = get_landmarks_from_object(person_landmarks_object)
    if landmarks is None:
        return False
    
    # Check for bending (existing logic)
    is_bending = detect_bending(person_landmarks_object)

    if not is_bending:
        return False

    try:
        # Key landmarks for theft detection (hands near torso/hips/bag area)
        # Using normalized coordinates, so distances are also normalized to frame size
        left_wrist = [landmarks[15].x, landmarks[15].y]
        right_wrist = [landmarks[16].x, landmarks[16].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        right_hip = [landmarks[24].x, landmarks[24].y]
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]

        # Heuristic 1: Both wrists are low (around or below hip level) and close to each other
        # This simulates hands going into a pocket or bag
        left_wrist_low = left_wrist[1] > left_hip[1] - 0.05 # Slightly above hip to below hip
        right_wrist_low = right_wrist[1] > right_hip[1] - 0.05
        
        # Distance between wrists (normalized)
        wrist_distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))

        # Heuristic 2: One wrist is below its corresponding shoulder and close to the body's center line
        # This simulates reaching into a front pocket or a bag held in front
        center_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
        
        left_wrist_below_shoulder = left_wrist[1] > left_shoulder[1]
        left_wrist_near_center = abs(left_wrist[0] - center_shoulder_x) < Config.THEFT_PROXIMITY_THRESHOLD
        
        right_wrist_below_shoulder = right_wrist[1] > right_shoulder[1]
        right_wrist_near_center = abs(right_wrist[0] - center_shoulder_x) < Config.THEFT_PROXIMITY_THRESHOLD

        # If bending, and one of these hand positions is met:
        if is_bending and (
            (left_wrist_low and right_wrist_low and wrist_distance < Config.THEFT_PROXIMITY_THRESHOLD * 2) or # Both hands low and close
            (left_wrist_below_shoulder and left_wrist_near_center) or # Left hand low and near center
            (right_wrist_below_shoulder and right_wrist_near_center) # Right hand low and near center
        ):
            return True
        
        return False

    except (IndexError, AttributeError) as e:
        print(f"Error in detect_theft: {e}. Landmarks object type inside function: {type(person_landmarks_object)}")
        return False
    """

def is_in_restricted_zone(bbox, zone=Config.RESTRICTED_ZONE):
    if not zone or len(zone) != 4:
        return False
    x1_person, y1_person, x2_person, y2_person = bbox
    center_x = (x1_person + x2_person) / 2
    center_y = (y1_person + y2_person) / 2
    zone_x1, zone_y1, zone_x2, zone_y2 = zone
    return (zone_x1 <= center_x <= zone_x2 and
            zone_y1 <= center_y <= zone_y2)

# Load YOLO model using Ultralytics
yolo_model = None
try:
    # Ultralytics loads .pt files directly.
    # Ensure Config.YOLO_MODEL_PATH points to your downloaded yolov3-tiny.pt
    yolo_model = YOLO(Config.YOLO_MODEL_PATH) 
    print("YOLO model (via Ultralytics) loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model with Ultralytics: {e}")
    print(f"Please ensure the YOLO model file '{Config.YOLO_MODEL_PATH}' exists and is a valid Ultralytics (.pt) model.")
    yolo_model = None

def iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


# Frame processing
def process_frame(frame, pose_detector, face_analyzer, yolo_model):
    global tracked_persons, next_person_id
    height, width, channels = frame.shape
    
    current_frame_index = monitoring_data["processing_frame_index"] + 1
    
    # Initialize counts for this frame based on tracked persons
    current_standing_count_frame = 0
    current_bending_count_frame = 0
    
    # List to store IDs of persons detected in the current frame
    person_ids_in_current_frame = set()

    # Draw restricted zone first
    if Config.RESTRICTED_ZONE:
        x1_zone, y1_zone, x2_zone, y2_zone = Config.RESTRICTED_ZONE
        cv2.rectangle(frame, (x1_zone, y1_zone), (x2_zone, y2_zone), (0, 255, 255), 2)
        # Reduced thickness for "Restricted Zone" label
        cv2.putText(frame, "Restricted Zone", (x1_zone, y1_zone - 10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) 

    # Perform YOLO detection using Ultralytics
    if yolo_model:
        results = yolo_model.predict(source=frame, conf=Config.YOLO_CONF_THRESHOLD, iou=Config.YOLO_NMS_THRESHOLD, classes=[0], verbose=False)
        
        if results and len(results) > 0:
            for r in results: # r is a Results object
                for box in r.boxes: # box is a Boxes object
                    # Extract bounding box coordinates in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Ensure it's a 'person' (class ID 0 in COCO)
                    if int(box.cls.item()) == 0: 
                        current_person_bbox = (x1, y1, x2, y2)
                        
                        # --- Tracking Logic ---
                        matched_person_id = None
                        best_iou = 0
                        
                        # Iterate through existing tracked persons to find a match
                        for p_id, p_data in tracked_persons.items():
                            track_bbox = p_data['bbox']
                            current_iou = iou(current_person_bbox, track_bbox)
                            if current_iou > Config.TRACKING_IOU_THRESHOLD and current_iou > best_iou:
                                best_iou = current_iou
                                matched_person_id = p_id
                        
                        if matched_person_id is None:
                            # New person detected
                            matched_person_id = next_person_id
                            next_person_id += 1
                            tracked_persons[matched_person_id] = {
                                'bbox': current_person_bbox,
                                'last_seen_frame': current_frame_index,
                                'is_unknown_face': False, # Default
                                'is_standing': False,
                                'is_bending': False,
                                'is_known_face': False, # Default
                                'face_logged': False # To prevent re-logging unknown faces
                            }
                            print(f"New person detected with ID: {matched_person_id}")
                        else:
                            # Update existing person's data
                            tracked_persons[matched_person_id]['bbox'] = current_person_bbox
                            tracked_persons[matched_person_id]['last_seen_frame'] = current_frame_index
                        
                        person_ids_in_current_frame.add(matched_person_id)

                        # Draw person bounding box
                        person_bbox_color = (200, 150, 0) # A more subdued blue/purple
                        cv2.rectangle(frame, (x1, y1), (x2, y2), person_bbox_color, 1) # Thinner line
                        cv2.putText(frame, f"Person {matched_person_id}", (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, person_bbox_color, 1) 
                        
                        # Crop person for MediaPipe and InsightFace
                        person_crop = frame[y1:y2, x1:x2]
                        if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                            continue # Skip empty crops

                        rgb_person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                        # --- InsightFace (Face Detection & Recognition) on person crop ---
                        if face_analyzer:
                            faces_in_crop = face_analyzer.get(rgb_person_crop)
                            
                            face_detected_for_person = False
                            for face in faces_in_crop:
                                face_detected_for_person = True
                                # Adjust face bbox to original frame coordinates
                                face_bbox = face.bbox.astype(np.int32)
                                fx1, fy1, fx2, fy2 = face_bbox[0], face_bbox[1], face_bbox[2], face_bbox[3]
                                
                                # Convert to original frame coordinates
                                original_fx1 = x1 + fx1
                                original_fy1 = y1 + fy1
                                original_fx2 = x1 + fx2
                                original_fy2 = y1 + fy2

                                emb = face.embedding
                                label = "Unknown"
                                color = (0, 165, 255) # Orange for unknown
                                
                                # Assume unknown initially
                                tracked_persons[matched_person_id]['is_unknown_face'] = True
                                tracked_persons[matched_person_id]['is_known_face'] = False

                                blacklisted_name = find_match(emb, blacklist_faces)
                                if blacklisted_name:
                                    label = f"Blacklisted: {blacklisted_name}"
                                    color = (0, 0, 255) # Red for blacklisted
                                    log_event("blacklist_alert", f"Blacklisted person {blacklisted_name} detected.")
                                    play_alert("restricted")
                                    # This person is blacklisted, not unknown or known in the general sense
                                    tracked_persons[matched_person_id]['is_unknown_face'] = False 
                                    tracked_persons[matched_person_id]['is_known_face'] = False

                                    if is_in_restricted_zone((original_fx1, original_fy1, original_fx2, original_fy2)):
                                        monitoring_data["restricted_zone_breached"] = True
                                        log_event("restricted_zone_breach", f"Blacklisted person {blacklisted_name} in restricted zone.")
                                else:
                                    known_name = find_match(emb, registered_faces)
                                    if known_name:
                                        label = f"Known: {known_name}"
                                        color = (0, 255, 0) # Green for known
                                        tracked_persons[matched_person_id]['is_known_face'] = True
                                        tracked_persons[matched_person_id]['is_unknown_face'] = False # Not unknown if known
                                    else:
                                        # It's an unknown face, and we haven't logged it for this person ID yet
                                        if not tracked_persons[matched_person_id]['face_logged']:
                                            log_event("unknown_face_detected", f"An unknown face (ID: {matched_person_id}) was detected.")
                                            face_image_crop = person_crop[fy1:fy2, fx1:fx2]
                                            if face_image_crop.shape[0] > 0 and face_image_crop.shape[1] > 0:
                                                log_unknown_face(face_image_crop, similarity_score=-1.0)
                                            tracked_persons[matched_person_id]['face_logged'] = True # Mark as logged
                                        # tracked_persons[matched_person_id]['is_unknown_face'] remains True as set above

                                # Draw face bounding box and label (less bold)
                                cv2.rectangle(frame, (original_fx1, original_fy1), (original_fx2, original_fy2), color, 2)
                                cv2.putText(frame, label, (original_fx1, original_fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1) # Thickness changed to 1
                            
                            if not face_detected_for_person:
                                # If a person is detected but no face, reset face status
                                tracked_persons[matched_person_id]['is_unknown_face'] = False
                                tracked_persons[matched_person_id]['is_known_face'] = False
                                tracked_persons[matched_person_id]['face_logged'] = False # Allow logging if a face appears later

                        # --- MediaPipe PoseLandmarker on person crop ---
                        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_person_crop)
                        pose_results_crop = pose_detector.detect(image_mp)
                        
                        # Reset posture for this person for current frame
                        tracked_persons[matched_person_id]['is_standing'] = False
                        tracked_persons[matched_person_id]['is_bending'] = False

                        if pose_results_crop.pose_landmarks:
                            for person_landmarks_object in pose_results_crop.pose_landmarks:
                                raw_landmarks_list = get_landmarks_from_object(person_landmarks_object)

                                if raw_landmarks_list:
                                    converted_landmarks = []
                                    for lm in raw_landmarks_list:
                                        # Convert landmark coordinates from crop to original frame
                                        converted_lm = NormalizedLandmark(
                                            x=(lm.x * (x2 - x1) + x1) / width, 
                                            y=(lm.y * (y2 - y1) + y1) / height, 
                                            z=lm.z
                                        )
                                        if hasattr(lm, 'visibility'):
                                            converted_lm.visibility = lm.visibility
                                        if hasattr(lm, 'presence'):
                                            converted_lm.presence = lm.presence
                                        converted_landmarks.append(converted_lm)
                                    
                                    landmarks_for_drawing = NormalizedLandmarkList(landmark=converted_landmarks)
                                    
                                    # Draw MediaPipe landmarks (adjusted thickness and radius)
                                    mp_drawing.draw_landmarks(
                                        image=frame, 
                                        landmark_list=landmarks_for_drawing, 
                                        connections=mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), # Reduced thickness and radius
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2) # Reduced thickness and radius
                                    )
                                
                                    # Posture detection based on the pose in the crop
                                    posture_text_y_offset = 0
                                    if detect_standing(person_landmarks_object):
                                        tracked_persons[matched_person_id]['is_standing'] = True
                                        cv2.putText(frame, "STANDING", (x1 + 5, y1 + 20 + posture_text_y_offset),
                                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) # Thickness changed to 1
                                        posture_text_y_offset += 20
                                    
                                    if detect_bending(person_landmarks_object):
                                        tracked_persons[matched_person_id]['is_bending'] = True
                                        cv2.putText(frame, "BENDING", (x1 + 5, y1 + 20 + posture_text_y_offset),
                                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1) # Thickness changed to 1
                                        posture_text_y_offset += 20

                                    # Theft detection logic is commented out as requested.
                                    # if detect_theft(person_landmarks_object):
                                    #     theft_detected_this_frame = True
                                    #     monitoring_data["theft_detected"] = True
                                    #     play_alert("theft") # This will now respect the cooldown
                                    #     cv2.putText(frame, "THEFT ALERT!", (x1 + 5, y1 + 20 + posture_text_y_offset),
                                    #                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    #     log_event("theft_alert", "Potential theft posture detected.")
    
    # --- Cleanup and Final Counting based on tracked_persons ---
    
    # Remove persons not seen for a while
    persons_to_remove = [
        p_id for p_id, p_data in tracked_persons.items() 
        if p_data['last_seen_frame'] < current_frame_index - Config.TRACKING_TIMEOUT_FRAMES
    ]
    for p_id in persons_to_remove:
        print(f"Person ID {p_id} timed out and removed from tracking.")
        del tracked_persons[p_id]

    # Recalculate counts based on currently tracked persons
    current_crowd_count = len(tracked_persons)
    current_standing_count = sum(1 for p_data in tracked_persons.values() if p_data['is_standing'])
    current_bending_count = sum(1 for p_data in tracked_persons.values() if p_data['is_bending'])
    total_unique_unknown_faces = sum(1 for p_data in tracked_persons.values() if p_data['is_unknown_face'])
    total_unique_known_faces = sum(1 for p_data in tracked_persons.values() if p_data['is_known_face'])

    monitoring_data.update({
        "current_crowd_count": current_crowd_count,
        "maximum_crowd_count": max(monitoring_data["maximum_crowd_count"], current_crowd_count),
        "current_standing_count": current_standing_count,
        "current_bending_count": current_bending_count,
        # These accumulated counts should be updated based on unique persons who exhibited the behavior
        # For simplicity for now, they reflect the current frame's unique counts.
        # If you need *total unique* standing/bending over time, you'd need to store that in tracked_persons.
        # For now, these are effectively resetting per frame based on current tracked people.
        # To accumulate, you would do:
        # "standing_detections": monitoring_data["standing_detections"] + (current_standing_count - prev_standing_count_for_accum),
        # But this requires more complex state. Sticking to current frame counts for now as per "apply the same tracking-based logic".
        "standing_detections": current_standing_count, # This now reflects unique standing persons in the current frame
        "bending_detections": current_bending_count,   # This now reflects unique bending persons in the current frame
        "unknown_faces": total_unique_unknown_faces, # This now reflects unique unknown faces currently tracked
        "known_faces": total_unique_known_faces,     # This now reflects unique known faces currently tracked
        "last_updated": datetime.now().isoformat(),
        "processing_frame_index": current_frame_index
    })
    
    # NEW: Crowd alert logic
    if monitoring_data["current_crowd_count"] > Config.CROWD_THRESHOLD:
        play_alert("crowd") # This will now respect the cooldown

    return frame

# Video processing thread
def video_processor(source, email_for_report_arg=None):
    global monitoring_data, tracked_persons, next_person_id

    cap = None
    monitoring_data["theft_detected"] = False
    monitoring_data["restricted_zone_breached"] = False
    monitoring_data["email_for_report"] = email_for_report_arg
    monitoring_data["video_processing_complete"] = False
    
    # Reset tracking state for new session
    tracked_persons = {}
    next_person_id = 0

    # Reset counts for new session
    monitoring_data["known_faces"] = 0
    monitoring_data["standing_detections"] = 0
    monitoring_data["bending_detections"] = 0
    monitoring_data["unknown_faces"] = 0
    monitoring_data["maximum_crowd_count"] = 0
    monitoring_data["current_crowd_count"] = 0
    monitoring_data["processing_frame_index"] = 0


    model_path = os.path.join("models", "pose_landmarker_full.task")
    if not os.path.exists(model_path):
        print(f"Error: PoseLandmarker model not found at {model_path}. Please ensure it's in the 'models' directory.")
        monitoring_data["status"] = "error"
        log_event("model_load_error", f"PoseLandmarker model not found at {model_path}")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=Config.NUM_POSES_PER_PERSON_CROP, # Use the new config variable
        min_pose_detection_confidence=Config.POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=0.05,
        min_tracking_confidence=0.05
    )
    pose_detector = None
    try:
        pose_detector = PoseLandmarker.create_from_options(options)
        print("PoseLandmarker model loaded successfully for multi-person detection.")
    except Exception as e:
        print(f"Error loading PoseLandmarker model: {e}")
        monitoring_data["status"] = "error"
        log_event("model_load_error", f"Failed to load PoseLandmarker model: {e}")
        return

    # Check if YOLO model is loaded (now using ultralytics model)
    if yolo_model is None:
        print("ERROR: YOLO model (Ultralytics) not loaded. Person detection will not work.")
        monitoring_data["status"] = "error"
        log_event("yolo_load_error", "YOLO model failed to load.")
        return

    try:
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error opening video source: {source}")
            log_event("video_error", f"Could not open video source {source}")
            monitoring_data["status"] = "error"
            return
            
        print(f"Successfully opened video source: {source}")
        monitoring_data["active_clients"] += 1
        monitoring_data["status"] = "monitoring"
        
        frame_count_total = 0 
        while monitoring_data["status"] == "monitoring":
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or monitoring stopped.")
                break 
            
            frame_count_total += 1
            
            if frame_count_total % Config.FRAME_SKIP != 0:
                continue

            # Pass ultralytics YOLO model to process_frame
            processed_frame = process_frame(frame, pose_detector, face_analysis, yolo_model)
            
            _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            monitoring_data["last_frame"] = buffer.tobytes()
            
            time.sleep(0.01) 
            
        if not ret: 
            monitoring_data["video_processing_complete"] = True
        else:
            monitoring_data["video_processing_complete"] = False

        if monitoring_data["email_for_report"] and monitoring_data["video_processing_complete"] and \
           (monitoring_data["restricted_zone_breached"] or monitoring_data["theft_detected"] or monitoring_data["unknown_faces"] > 0):
            
            final_report_subject = "CrowdVision AI Final Report: Alerts Detected"
            final_report_body = f"The video analysis has completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
            final_report_body += "Here's a summary of the detected events:\n\n"
            
            if monitoring_data["theft_detected"]:
                final_report_body += "- **Potential theft detected.**\n"
            if monitoring_data["restricted_zone_breached"]:
                final_report_body += "- **Restricted zone breached.**\n"
            if monitoring_data["unknown_faces"] > 0:
                final_report_body += f"- **{monitoring_data['unknown_faces']} unique unknown faces detected.**\n"
            
            final_report_body += f"\nTotal unique standing detections: {monitoring_data['standing_detections']}\n"
            final_report_body += f"Total unique bending detections: {monitoring_data['bending_detections']}\n"
            final_report_body += f"Maximum crowd count: {monitoring_data['maximum_crowd_count']}\n"
            
            send_email_with_summary(monitoring_data["email_for_report"], final_report_subject, final_report_body)
        elif monitoring_data["email_for_report"] and monitoring_data["video_processing_complete"]:
             print("Video completed, but no significant alerts to report via email.")
        else:
            print("Email report not sent: either no email provided, video was manually stopped, or no alerts occurred.")

    except Exception as e:
        print(f"Video processing error: {str(e)}")
        log_event("processing_error", str(e))
        monitoring_data["status"] = "error"
    finally:
        if cap is not None:
            cap.release()
        if pose_detector:
            pose_detector.close()
        
        if monitoring_data["current_temp_video_path"] and os.path.exists(monitoring_data["current_temp_video_path"]):
            try:
                os.remove(monitoring_data["current_temp_video_path"])
                print(f"Deleted temporary video file: {monitoring_data['current_temp_video_path']}")
            except Exception as e:
                print(f"Error deleting temporary video file {monitoring_data['current_temp_video_path']}: {e}")
            monitoring_data["current_temp_video_path"] = None

        if monitoring_data["status"] == "monitoring":
            monitoring_data["status"] = "completed"
        elif monitoring_data["status"] == "stopping":
             monitoring_data["status"] = "ready"
        elif monitoring_data["status"] == "error":
             pass
        
        monitoring_data["active_clients"] = max(0, monitoring_data["active_clients"] - 1)
        
        monitoring_data["email_for_report"] = None 
        monitoring_data["last_frame"] = None 
        print("Video processing thread finished.")


# Flask routes
@app.route('/')
def home():
    return jsonify({
        "message": "CrowdVision AI Backend",
        "status": "running",
        "endpoints": {
            "GET /video_feed": "Live video stream with annotations",
            "POST /start": "Start monitoring (accepts video_path, email_for_report)",
            "GET /status": "Get current monitoring status and statistics",
            "POST /stop": "Stop monitoring",
            "POST /upload_video": "Upload a video file for temporary processing"
        }
    })

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if monitoring_data["last_frame"] is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       monitoring_data["last_frame"] + b'\r\n')
            else:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                status_text = "Waiting for video feed..."
                if monitoring_data["status"] == "ready":
                    status_text = "Ready to start monitoring."
                elif monitoring_data["status"] == "error":
                    status_text = "Monitoring Error. Check backend logs."
                elif monitoring_data["status"] == "completed":
                    status_text = "Video analysis completed."

                cv2.putText(blank_frame, status_text, 
                                 (max(0, (blank_frame.shape[1] - cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]) // 2), 
                                  blank_frame.shape[0] // 2), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                _, buffer = cv2.imencode('.jpg', blank_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
    
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file part in the request"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        original_ext = file.filename.rsplit('.', 1)[1].lower()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{original_ext}")
        temp_file_path = temp_file.name
        temp_file.close()

        try:
            file.save(temp_file_path)
            print(f"File uploaded and temporarily saved to: {temp_file_path}")
            
            monitoring_data["current_temp_video_path"] = temp_file_path 

            return jsonify({
                "status": "success",
                "message": "Video uploaded successfully to temporary location",
                "server_video_path": temp_file_path
            }), 200
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return jsonify({"status": "error", "message": f"Failed to save temporary file: {str(e)}"}), 500
    else:
        return jsonify({
            "status": "error", 
            "message": "File type not allowed. Allowed types are: " + ', '.join(Config.ALLOWED_EXTENSIONS)
        }), 400


@app.route('/start', methods=['POST'])
def start_monitoring():
    global tracked_persons, next_person_id
    try:
        data = request.get_json()
        
        if monitoring_data["status"] == "monitoring":
            return jsonify({
                "status": "warning",
                "message": "Monitoring is already active. Please stop it first if you want to restart."
            }), 409
        
        # Reset tracking state for new session
        tracked_persons = {}
        next_person_id = 0

        # Reset current counts when starting new monitoring session
        monitoring_data.update({
            "maximum_crowd_count": 0,
            "current_crowd_count": 0,
            "current_standing_count": 0, # Reset for new session
            "current_bending_count": 0,  # Reset for new session
            "standing_detections": 0,    # Reset total accumulated counts for new session
            "bending_detections": 0,     # Reset total accumulated counts for new session
            "unknown_faces": 0,
            "known_faces": 0,
            "theft_detected": False,
            "restricted_zone_breached": False,
            "status": "starting",
            "last_updated": None,
            "last_frame": None,
            "frame_counter": 0,
            "processing_frame_index": 0,
            "video_processing_complete": False,
            "current_temp_video_path": None
        })
        
        video_path_from_request = data.get("video_path")
        email_to_send_report = data.get("email_for_report")

        video_source_to_use = None
        
        if video_path_from_request == monitoring_data["current_temp_video_path"] and monitoring_data["current_temp_video_path"] is not None:
            video_source_to_use = monitoring_data["current_temp_video_path"]
            print(f"Using uploaded temporary video source: {video_source_to_use}")
        elif video_path_from_request:
            if video_path_from_request.lower() == "webcam":
                video_source_to_use = 0
                print("Using webcam as video source.")
            elif os.path.exists(video_path_from_request):
                video_source_to_use = video_path_from_request
                print(f"Using file video source: {video_source_to_use}")
            else:
                full_video_path = os.path.join(Config.VIDEO_FOLDER, video_path_from_request)
                if os.path.exists(full_video_path):
                    video_source_to_use = full_video_path
                    print(f"Using file video source from default folder: {full_video_path}")
                else:
                    return jsonify({
                        "status": "error",
                        "message": f"Video file '{video_path_from_request}' not found. "
                                   "Ensure it's a valid absolute path or a filename in the 'video' folder, "
                                   "or use the /upload_video endpoint first to get a temporary path."
                    }), 400
        else:
            video_source_to_use = os.path.join(Config.VIDEO_FOLDER, Config.DEFAULT_VIDEO)
            if not os.path.exists(video_source_to_use):
                 return jsonify({
                     "status": "error",
                     "message": f"Default video file '{video_source_to_use}' not found. Please ensure it exists."
                 }), 400
            print(f"No video_path provided, using default: {video_source_to_use}")

        threading.Thread(
            target=video_processor,
            args=(video_source_to_use, email_to_send_report),
            daemon=True
        ).start()
        
        return jsonify({
            "status": "success",
            "message": "Monitoring started",
            "data": {k: v for k, v in monitoring_data.items() if k != "last_frame"}
        })
    except Exception as e:
        monitoring_data["status"] = "error"
        log_event("start_error", str(e))
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/stop', methods=['POST'])
def stop_monitoring():
    global monitoring_data, tracked_persons, next_person_id
    if monitoring_data["status"] == "ready":
        return jsonify({
            "status": "info",
            "message": "Monitoring is not active."
        }), 200

    monitoring_data["status"] = "stopping"
    log_event("monitoring_stop", "Monitoring requested to stop by user")
    
    time.sleep(1) 
    
    # Reset tracking state on stop
    tracked_persons = {}
    next_person_id = 0

    monitoring_data.update({
        "status": "ready",
        "maximum_crowd_count": 0,
        "current_crowd_count": 0,
        "current_standing_count": 0, # Reset for stop
        "current_bending_count": 0,  # Reset for stop
        "standing_detections": 0,    # Reset total accumulated counts for stop
        "bending_detections": 0,     # Reset total accumulated counts for stop
        "unknown_faces": 0,
        "known_faces": 0,
        "theft_detected": False,
        "restricted_zone_breached": False,
        "last_updated": None,
        "last_frame": None,
        "frame_counter": 0,
        "processing_frame_index": 0,
        "email_for_report": None,
        "video_processing_complete": False,
        "current_temp_video_path": None
    })

    return jsonify({
        "status": "success",
        "message": "Monitoring stopped."
    })

@app.route('/status')
def get_status():
    # Return current_standing_count and current_bending_count for per-frame display
    return jsonify({k: v for k, v in monitoring_data.items() if k != "last_frame"})

def send_email_with_frame(frame, email_to, subject="CrowdVision AI Alert"):
    if not email_to:
        print("No recipient email provided for alert.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = Config.EMAIL_SENDER
        msg['To'] = email_to
        msg['Subject'] = subject

        body = f"An alert from CrowdVision AI System has been triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. See attached frame."
        msg.attach(MIMEText(body, 'plain'))

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            img_data = buffer.tobytes()
            image = MIMEImage(img_data, name=f'alert_frame_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
            image.add_header('Content-Disposition', 'attachment', filename=image.get_param('name'))
            image.add_header('Content-ID', '<alert_frame>') # Important for embedding in HTML emails
            msg.attach(image)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(Config.EMAIL_SENDER, Config.EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Email alert sent to {email_to} with subject '{subject}'.")
        log_event("email_alert_sent", f"Email alert sent to {email_to}.")
    except Exception as e:
        print(f"Failed to send email alert to {email_to}: {e}")
        log_event("email_alert_error", f"Failed to send email alert to {email_to}: {e}")

def send_email_with_summary(email_to, subject="CrowdVision AI Report", body=""):
    if not email_to:
        print("No recipient email provided for summary report.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = Config.EMAIL_SENDER
        msg['To'] = email_to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Get unknown face image paths from the database
        unknown_face_image_paths = get_unknown_face_image_paths_from_db()

        # Attach each unknown face image
        for i, img_path in enumerate(unknown_face_image_paths):
            try:
                with open(img_path, 'rb') as img_file:
                    img_data = img_file.read()
                image = MIMEImage(img_data, name=f'unknown_face_{i+1}.jpg')
                image.add_header('Content-Disposition', 'attachment', filename=f'unknown_face_{i+1}.jpg')
                msg.attach(image)
                print(f"Attached unknown face image: {img_path}")
            except FileNotFoundError:
                print(f"Warning: Unknown face image file not found: {img_path}")
            except Exception as e:
                print(f"Error attaching image {img_path}: {e}")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(Config.EMAIL_SENDER, Config.EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Summary email report sent to {email_to} with subject '{subject}'.")
        log_event("email_summary_sent", f"Summary email report sent to {email_to}.")
    except Exception as e:
        print(f"Failed to send summary email to {email_to}: {e}")
        log_event("email_summary_error", f"Failed to send summary email to {email_to}: {e}")
    finally:
        # Clear the unknown faces table after sending the report
        clear_unknown_faces_db()


# Global variables to track last alert times
last_theft_alert_time = 0
last_crowd_alert_time = 0

def play_alert(sound_type):
    global last_theft_alert_time, last_crowd_alert_time
    current_time = time.time()
    
    # Determine which alert's timer to check
    if sound_type == "theft":
        last_alert_time_ref = last_theft_alert_time
    elif sound_type == "crowd":
        last_alert_time_ref = last_crowd_alert_time
    elif sound_type == "restricted": # Keep existing behavior for restricted zone, no specific cooldown for this one
        last_alert_time_ref = 0 # This ensures it can play more frequently if needed, or if not busy
    else:
        print(f"Unknown sound type: {sound_type}")
        return

    if not pygame_mixer_initialized:
        print("Pygame mixer not initialized. Cannot play audio alerts.")
        return

    # Check if enough time has passed since the last alert of this type
    # For 'theft' and 'crowd', enforce 10-second cooldown
    if (sound_type == "theft" or sound_type == "crowd") and (current_time - last_alert_time_ref < 10):
        print(f"Audio alert {sound_type} skipped, still in cooldown period.")
        return

    try:
        sounds = {
            "crowd": os.path.join(Config.AUDIO_FOLDER, "crowd.mp3"),
            "theft": os.path.join(Config.AUDIO_FOLDER, "theft.mp3"),
            "restricted": os.path.join(Config.AUDIO_FOLDER, "restricted.mp3")
        }
        audio_file_path = sounds.get(sound_type)
        
        if audio_file_path and os.path.exists(audio_file_path):
            # Only play if no music is currently playing
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(audio_file_path)
                pygame.mixer.music.play()
                print(f"Playing audio alert: {sound_type}")
                
                # Update the last alert time for the specific type AFTER playing
                if sound_type == "theft":
                    last_theft_alert_time = current_time
                elif sound_type == "crowd":
                    last_crowd_alert_time = current_time
                
                # Set a timer to stop the music after 2 seconds
                threading.Timer(2.0, pygame.mixer.music.stop).start()
                
            else:
                print(f"Audio alert {sound_type} skipped, another alert is currently playing.")
        else:
            print(f"Audio file not found for type: {sound_type} at {audio_file_path}")
    except Exception as e:
        print(f"Audio error playing {sound_type}: {e}")
        log_event("audio_error", f"Failed to play audio for {sound_type}: {e}")

if __name__ == '__main__':
    os.makedirs(Config.AUDIO_FOLDER, exist_ok=True)
    os.makedirs(Config.VIDEO_FOLDER, exist_ok=True)
    os.makedirs("models", exist_ok=True) # Ensure base models directory exists
    os.makedirs(Config.YOLO_MODEL_DIR, exist_ok=True) # Ensure YOLO models directory exists
    os.makedirs(Config.UNKNOWN_FACES_FOLDER, exist_ok=True) # Ensure unknown_faces directory exists

    init_db()

    print("\n--- CrowdVision AI Server Starting ---")
    print("Access the API at: http://127.0.0.1:8000/")
    print("Available video sources:")
    print(f"- Default video file: '{os.path.join(Config.VIDEO_FOLDER, Config.DEFAULT_VIDEO)}'")
    print("- To use webcam, set 'video_path' to 'webcam' in the POST /start request.")
    print("- To use an uploaded video, first POST the file to /upload_video, then use the returned path with /start.")
    print("\nIMPORTANT: For person detection, ensure YOLOv3 model files (yolov3.cfg, yolov3.weights, coco.names) are in the 'models/yolo' directory.")
    print("You can use 'yolov3-tiny' versions for faster performance if needed.")

    print("\nStarting server with Waitress...")
    serve(app, host='0.0.0.0', port=8000)
