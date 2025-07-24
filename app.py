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
    DEFAULT_VIDEO = "theft.mp4" # Or standing.webm, depending on your default
    STANDING_THRESHOLD = 160
    BENDING_THRESHOLD = 150
    
    # Loosening face match threshold even further (be cautious of false positives)
    FACE_MATCH_THRESHOLD = 0.9 # Higher value means less strict matching
    
    CROWD_THRESHOLD = 5 
    
    # Lowering pose detection confidence significantly
    POSE_DETECTION_CONFIDENCE = 0.05 # Very aggressive
    
    # Corrected RESTRICTED_ZONE to have a proper width/height
    RESTRICTED_ZONE = (280, 50, 360, 430) # (x1, y1, x2, y2)
    FRAME_SKIP = 60
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


# Global monitoring data
monitoring_data = {
    "status": "ready", 
    "maximum_crowd_count": 0,
    "current_crowd_count": 0,
    "current_standing_count": 0, # NEW: Per-frame standing count
    "current_bending_count": 0,  # NEW: Per-frame bending count
    "standing_detections": 0,    # Total accumulated standing detections
    "bending_detections": 0,     # Total accumulated bending detections
    "unknown_faces": 0,
    "known_faces": 0,
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
            cur.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(50) NOT NULL,
                description TEXT,
                timestamp TIMESTAMP NOT NULL
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
        print(f"Error in detect_standing: {e}. Landmarks object type inside function: {type(person_landmarks_object)}")
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
        print(f"Error in detect_bending: {e}. Landmarks object type inside function: {type(person_landmarks_object)}")
        return False

def detect_theft(person_landmarks_object):
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


# Frame processing
def process_frame(frame, pose_detector, face_analyzer, yolo_model): # Removed yolo_classes as not directly used by ultralytics predict
    height, width, channels = frame.shape
    
    # Reset counts for this frame
    face_count_this_frame = 0
    standing_this_frame = 0
    bending_this_frame = 0
    unknown_faces_this_frame = 0
    known_faces_this_frame = 0 
    theft_detected_this_frame = False
    restricted_zone_breach_this_frame = False

    # Draw restricted zone first
    if Config.RESTRICTED_ZONE:
        x1_zone, y1_zone, x2_zone, y2_zone = Config.RESTRICTED_ZONE
        cv2.rectangle(frame, (x1_zone, y1_zone), (x2_zone, y2_zone), (0, 255, 255), 2)
        cv2.putText(frame, "Restricted Zone", (x1_zone, y1_zone - 10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Perform YOLO detection using Ultralytics
    yolo_person_count = 0
    if yolo_model:
        # Predict on the current frame. classes=[0] filters for 'person' class in COCO dataset.
        # verbose=False to suppress extensive logging from ultralytics
        results = yolo_model.predict(source=frame, conf=Config.YOLO_CONF_THRESHOLD, iou=Config.YOLO_NMS_THRESHOLD, classes=[0], verbose=False)
        
        if results and len(results) > 0:
            for r in results: # r is a Results object
                for box in r.boxes: # box is a Boxes object
                    # Extract bounding box coordinates in xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf.item() # Confidence score
                    cls = int(box.cls.item()) # Class ID

                    # Ensure it's a 'person' (class ID 0 in COCO)
                    if cls == 0: 
                        yolo_person_count += 1

                        # Draw person bounding box (lighter color, thinner line)
                        person_bbox_color = (200, 150, 0) # A more subdued blue/purple
                        cv2.rectangle(frame, (x1, y1), (x2, y2), person_bbox_color, 1) # Thinner line
                        cv2.putText(frame, f"Person", (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, person_bbox_color, 1) # Smaller text, thinner line
                        
                        # Crop person for MediaPipe and InsightFace
                        person_crop = frame[y1:y2, x1:x2]
                        if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                            continue # Skip empty crops

                        rgb_person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                        # --- InsightFace (Face Detection & Recognition) on person crop ---
                        if face_analyzer:
                            faces_in_crop = face_analyzer.get(rgb_person_crop)
                            
                            for face in faces_in_crop:
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

                                blacklisted_name = find_match(emb, blacklist_faces)
                                if blacklisted_name:
                                    label = f"Blacklisted: {blacklisted_name}"
                                    color = (0, 0, 255) # Red for blacklisted
                                    log_event("blacklist_alert", f"Blacklisted person {blacklisted_name} detected.")
                                    play_alert("restricted")

                                    if is_in_restricted_zone((original_fx1, original_fy1, original_fx2, original_fy2)):
                                        restricted_zone_breach_this_frame = True
                                        monitoring_data["restricted_zone_breached"] = True
                                        log_event("restricted_zone_breach", f"Blacklisted person {blacklisted_name} in restricted zone.")
                                else:
                                    known_name = find_match(emb, registered_faces)
                                    if known_name:
                                        label = f"Known: {known_name}"
                                        color = (0, 255, 0) # Green for known
                                        known_faces_this_frame += 1
                                    else:
                                        unknown_faces_this_frame += 1
                                        log_event("unknown_face_detected", "An unknown face was detected.")

                                # Draw face bounding box and label (more prominent than person bbox)
                                cv2.rectangle(frame, (original_fx1, original_fy1), (original_fx2, original_fy2), color, 2)
                                cv2.putText(frame, label, (original_fx1, original_fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                face_count_this_frame += 1 # Count faces detected within person crops

                        # --- MediaPipe PoseLandmarker on person crop ---
                        # It's crucial to pass the *cropped* image to MediaPipe for better focus
                        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_person_crop)
                        pose_results_crop = pose_detector.detect(image_mp)
                        
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
                                    # Draw posture text near the person's head/shoulder for clarity
                                    posture_text_y_offset = 0
                                    if detect_standing(person_landmarks_object):
                                        standing_this_frame += 1
                                        cv2.putText(frame, "STANDING", (x1 + 5, y1 + 20 + posture_text_y_offset),
                                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        posture_text_y_offset += 20
                                    
                                    if detect_bending(person_landmarks_object):
                                        bending_this_frame += 1
                                        cv2.putText(frame, "BENDING", (x1 + 5, y1 + 20 + posture_text_y_offset),
                                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                                        posture_text_y_offset += 20

                                    if detect_theft(person_landmarks_object):
                                        theft_detected_this_frame = True
                                        monitoring_data["theft_detected"] = True
                                        play_alert("theft") # This will now respect the cooldown
                                        cv2.putText(frame, "THEFT ALERT!", (x1 + 5, y1 + 20 + posture_text_y_offset),
                                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        log_event("theft_alert", "Potential theft posture detected.")

    monitoring_data.update({
        "current_crowd_count": yolo_person_count, # This is now the count from YOLO (persons detected)
        "maximum_crowd_count": max(monitoring_data["maximum_crowd_count"], yolo_person_count),
        "current_standing_count": standing_this_frame, # NEW: Update per-frame standing count
        "current_bending_count": bending_this_frame,   # NEW: Update per-frame bending count
        "standing_detections": monitoring_data["standing_detections"] + standing_this_frame, # Still accumulate total
        "bending_detections": monitoring_data["bending_detections"] + bending_this_frame,   # Still accumulate total
        "unknown_faces": monitoring_data["unknown_faces"] + unknown_faces_this_frame,
        "known_faces": monitoring_data["known_faces"] + known_faces_this_frame,
        "last_updated": datetime.now().isoformat(),
        "processing_frame_index": monitoring_data["processing_frame_index"] + 1
    })
    
    # NEW: Crowd alert logic
    if monitoring_data["current_crowd_count"] > Config.CROWD_THRESHOLD:
        play_alert("crowd") # This will now respect the cooldown

    return frame

# Video processing thread
def video_processor(source, email_for_report_arg=None):
    global monitoring_data

    cap = None
    monitoring_data["theft_detected"] = False
    monitoring_data["restricted_zone_breached"] = False
    monitoring_data["email_for_report"] = email_for_report_arg
    monitoring_data["video_processing_complete"] = False
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
            
            frame_count_total += 2
            
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
                final_report_body += f"- **{monitoring_data['unknown_faces']} unknown faces detected.**\n"
            
            final_report_body += f"\nTotal standing detections: {monitoring_data['standing_detections']}\n"
            final_report_body += f"Total bending detections: {monitoring_data['bending_detections']}\n"
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
    try:
        data = request.get_json()
        
        if monitoring_data["status"] == "monitoring":
            return jsonify({
                "status": "warning",
                "message": "Monitoring is already active. Please stop it first if you want to restart."
            }), 409
        
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
    global monitoring_data
    if monitoring_data["status"] == "ready":
        return jsonify({
            "status": "info",
            "message": "Monitoring is not active."
        }), 200

    monitoring_data["status"] = "stopping"
    log_event("monitoring_stop", "Monitoring requested to stop by user")
    
    time.sleep(1) 
    
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

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(Config.EMAIL_SENDER, Config.EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Summary email report sent to {email_to} with subject '{subject}'.")
        log_event("email_summary_sent", f"Summary email report sent to {email_to}.")
    except Exception as e:
        print(f"Failed to send summary email to {email_to}: {e}")
        log_event("email_summary_error", f"Failed to send summary email to {email_to}: {e}")

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

