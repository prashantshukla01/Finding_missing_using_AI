
import cv2
import numpy as np
import logging
import insightface
from insightface.app import FaceAnalysis
import torch
import torch.nn.functional as F
from app.utils.augmentations import get_augmentations
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedFaceMatcher:
    def __init__(self, model_name='buffalo_l', det_size=(640, 640)):
        self.model_name = model_name
        self.det_size = det_size
        # ArcFace thresholds: 0.5 is usually a good balance for buffalo_l
        self.similarity_threshold = 0.5 
        self.quality_threshold = 0.65
        self.augmentations = get_augmentations()
        
        # Initialize InsightFace (Resulting in faster, single-pass detection)
        try:
            self.insight_app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            self.insight_app.prepare(ctx_id=0, det_size=det_size)
            logger.info("InsightFace model loaded successfully (Standardized on Buffalo_L)")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise
        
        # Detection results storage
        self.recent_detections = []
        self.found_persons = set()
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def extract_embeddings(self, image_path):
        """Extract face embeddings using InsightFace"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return None
            
            # Detect faces
            faces = self.insight_app.get(image)
            
            if len(faces) == 0:
                logger.warning(f"No faces detected in {image_path}")
                return None
            
            # Get the largest face (assuming the subject is the main focus)
            # Sorting by bounding box area is usually safer than det_score for "main subject"
            best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            
            embedding_data = {
                'insightface': best_face.embedding,
                'det_score': best_face.det_score,
                'bbox': best_face.bbox,
                'source': 'insightface',
                'all_detections': len(faces),
                'kps': best_face.kps if hasattr(best_face, 'kps') else None
            }
            
            logger.info(f"✅ Extracted embedding. Score: {best_face.det_score:.3f}")
            return embedding_data
            
        except Exception as e:
            logger.error(f"❌ Error extracting embeddings from {image_path}: {e}")
            return None
    
    def compare_embeddings(self, embedding1, embedding2):
        """Compare two face embeddings using Cosine Similarity"""
        if embedding1 is None or embedding2 is None:
            return 0.0, "INVALID"
        
        try:
            # We strictly expect 'insightface' keys now
            # Robustness: Handle if embedding is already a list/array (legacy or raw vector)
            if isinstance(embedding1, dict):
                vec1 = embedding1.get('insightface')
            else:
                vec1 = embedding1

            if isinstance(embedding2, dict):
                vec2 = embedding2.get('insightface')
            else:
                vec2 = embedding2

            if vec1 is None or vec2 is None:
                # Debug logging
                # logger.warning(f"Missing insightface vector. Keys 1: {embedding1.keys() if isinstance(embedding1, dict) else 'List'}, Keys 2: {embedding2.keys() if isinstance(embedding2, dict) else 'List'}")
                return 0.0, "INVALID_VECTOR"
            
            # Ensure numpy arrays
            if isinstance(vec1, list): vec1 = np.array(vec1)
            if isinstance(vec2, list): vec2 = np.array(vec2)
            
            # Standard Cosine Similarity
            # Sim = (A . B) / (||A|| * ||B||)
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            
            similarity = dot_product / (norm_a * norm_b + 1e-6)
            
            # Map similarity to confidence levels
            if similarity > 0.6:
                confidence = "EXCELLENT"
            elif similarity > 0.5:
                confidence = "HIGH"
            elif similarity > 0.4:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            return float(similarity), confidence
                
        except Exception as e:
            logger.error(f"❌ Error comparing embeddings: {e}")
            return 0.0, "ERROR"
    
    def validate_face_quality(self, embedding_data):
        """Validate if detected face meets quality standards"""
        if embedding_data is None:
            return False, "No embedding data"
        
        det_score = embedding_data.get('det_score', 0)
        
        if det_score < self.quality_threshold:
            return False, f"Low detection score: {det_score:.3f} (Threshold: {self.quality_threshold})"
        
        return True, "OK"
    
    def detect_and_match_faces_realtime(self, frame, lost_person_encodings, lost_person_names, threshold=0.5):
        """Real-time face detection and matching"""
        try:
            # InsightFace expects RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Assuming input is BGR (OpenCV default), convert to RGB
                # But check if caller already converted? Usually cv2 frames are BGR.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            detected_faces = self.insight_app.get(frame_rgb)
            
            matches = []
            
            for face in detected_faces:
                # Default match info
                match_info = {
                    'name': 'Unknown',
                    'similarity': 0.0,
                    'confidence': 'NONE',
                    'bbox': face.bbox,
                    'algorithm': 'insightface',
                    'found': False
                }

                # Compare with all lost persons
                best_sim = -1.0
                best_idx = -1
                
                # Check known faces
                for i, lost_embedding in enumerate(lost_person_encodings):
                    # Direct cosine similarity calculation here for speed
                    # (Avoid function call overhead in inner loop if possible, or just call helper)
                    # Let's call helper for consistency, it's fast enough
                    
                    # Fix: lost_embedding is now a direct numpy array from our unpacked list
                    sim, conf = self.compare_embeddings(
                        {'insightface': face.embedding},
                        {'insightface': lost_embedding}
                    )
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = i
                
                # Check threshold
                if best_idx != -1 and best_sim >= threshold:
                    match_info['name'] = lost_person_names[best_idx]
                    match_info['similarity'] = best_sim
                    match_info['confidence'] = "MATCH" # Simplified confidence
                    match_info['found'] = True
                    match_info['timestamp'] = datetime.now()
                    
                    # Log event
                    logger.info(f"🎯 FOUND: {match_info['name']} (Sim: {best_sim:.2f})")
                    
                    self.found_persons.add(match_info['name'])
                    self.recent_detections.append(match_info)
                    
                    # Keep history manageable
                    if len(self.recent_detections) > 100:
                        self.recent_detections = self.recent_detections[-100:]

                matches.append(match_info)
            
            return matches
            
        except Exception as e:
            logger.error(f"❌ Realtime detection error: {e}")
            return []

    def get_recent_detections(self, limit=10):
        """Get recent face detections with timestamps"""
        recent = self.recent_detections[-limit:]
        
        # Format for display
        formatted_detections = []
        for detection in recent:
            if 'timestamp' not in detection: 
                continue # Skip if no timestamp
                
            formatted_detections.append({
                'name': detection['name'],
                'similarity': f"{detection['similarity']:.3f}",
                'confidence': detection['confidence'],
                'timestamp': detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'time_ago': self.get_time_ago(detection['timestamp']),
                'stream_name': detection.get('stream_name', 'Unknown')
            })
        
        return formatted_detections
    
    def get_time_ago(self, timestamp):
        """Get human-readable time ago"""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"
    
    def clear_found_persons(self):
        self.found_persons.clear()