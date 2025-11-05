import cv2
import numpy as np
import logging
import insightface
from insightface.app import FaceAnalysis
from utils.augmentations import get_augmentations

# Try to import optional libraries
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedFaceMatcher:
    def __init__(self, model_name='buffalo_l', det_size=(640, 640)):
        self.model_name = model_name
        self.det_size = det_size
        self.similarity_threshold = 0.6
        self.quality_threshold = 0.7
        self.augmentations = get_augmentations()
        
        # Initialize multiple face detection algorithms
        try:
            self.insight_app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            self.insight_app.prepare(ctx_id=0, det_size=det_size)
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise
        
        # Initialize MTCNN for enhanced face detection
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNN()
                logger.info("MTCNN model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load MTCNN model: {e}")
                self.mtcnn = None
        else:
            self.mtcnn = None
        
        # Initialize dlib face detector and predictor
        if DLIB_AVAILABLE:
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
                self.dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                logger.info("Dlib models loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load dlib models: {e}")
                self.dlib_detector = None
                self.dlib_predictor = None
        else:
            self.dlib_detector = None
            self.dlib_predictor = None
        
        # Initialize face_recognition library
        if FACE_RECOGNITION_AVAILABLE:
            try:
                logger.info("Face recognition library initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize face recognition library: {e}")
        else:
            logger.warning("Face recognition library not available")
        
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
    
    def detect_faces_multi_algorithm(self, image):
        """Detect faces using multiple algorithms for maximum accuracy"""
        all_faces = []
        
        # 1. InsightFace detection
        try:
            insight_faces = self.insight_app.get(image)
            for face in insight_faces:
                all_faces.append({
                    'bbox': face.bbox,
                    'confidence': face.det_score,
                    'landmarks': face.kps if hasattr(face, 'kps') else None,
                    'embedding': face.embedding,
                    'algorithm': 'insightface'
                })
        except Exception as e:
            logger.warning(f"InsightFace detection failed: {e}")
        
        # 2. MTCNN detection (if available)
        if self.mtcnn:
            try:
                mtcnn_faces = self.mtcnn.detect_faces(image)
                for face in mtcnn_faces:
                    if face['confidence'] > 0.9:  # High confidence only
                        all_faces.append({
                            'bbox': [face['box'][0], face['box'][1], 
                                   face['box'][0] + face['box'][2], 
                                   face['box'][1] + face['box'][3]],
                            'confidence': face['confidence'],
                            'landmarks': face['keypoints'] if 'keypoints' in face else None,
                            'algorithm': 'mtcnn'
                        })
            except Exception as e:
                logger.warning(f"MTCNN detection failed: {e}")
        
        # 3. Dlib detection (if available)
        if self.dlib_detector:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                dlib_faces = self.dlib_detector(gray)
                for face in dlib_faces:
                    all_faces.append({
                        'bbox': [face.left(), face.top(), face.right(), face.bottom()],
                        'confidence': 0.8,  # Dlib doesn't provide confidence scores
                        'algorithm': 'dlib'
                    })
            except Exception as e:
                logger.warning(f"Dlib detection failed: {e}")
        
        # 4. Face_recognition library detection (if available)
        if FACE_RECOGNITION_AVAILABLE:
            try:
                face_locations = face_recognition.face_locations(image)
                for (top, right, bottom, left) in face_locations:
                    all_faces.append({
                        'bbox': [left, top, right, bottom],
                        'confidence': 0.75,  # Default confidence
                        'algorithm': 'face_recognition'
                    })
            except Exception as e:
                logger.warning(f"Face_recognition detection failed: {e}")
        
        return all_faces
    
    def extract_embeddings(self, image_path):
        """Extract face embeddings from image using multi-algorithm approach"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return None
            
            # Multi-algorithm face detection
            all_faces = self.detect_faces_multi_algorithm(image)
            
            if len(all_faces) == 0:
                logger.warning(f"No faces detected in {image_path}")
                return None
            
            # Get the best face (highest confidence score)
            best_face = max(all_faces, key=lambda x: x['confidence'])
            
            # If InsightFace found the face, use its embedding
            if best_face['algorithm'] == 'insightface':
                embedding = best_face['embedding']
            else:
                # For other algorithms, extract embedding using InsightFace on the cropped face
                x1, y1, x2, y2 = map(int, best_face['bbox'])
                cropped_face = image[y1:y2, x1:x2]
                
                if cropped_face.size > 0:
                    insight_faces = self.insight_app.get(cropped_face)
                    if len(insight_faces) > 0:
                        embedding = insight_faces[0].embedding
                    else:
                        logger.warning(f"Could not extract embedding from cropped face")
                        return None
                else:
                    logger.warning(f"Invalid face crop")
                    return None
            
            embedding_data = {
                'insightface': embedding,
                'det_score': best_face['confidence'],
                'bbox': best_face['bbox'],
                'source': best_face['algorithm'],
                'all_detections': len(all_faces)
            }
            
            # Add landmarks if available
            if 'landmarks' in best_face and best_face['landmarks'] is not None:
                embedding_data['landmarks'] = best_face['landmarks']
            
            logger.info(f"✅ Successfully extracted embeddings from {image_path} "
                       f"(algorithm: {best_face['algorithm']}, confidence: {best_face['confidence']:.3f})")
            return embedding_data
            
        except Exception as e:
            logger.error(f"❌ Error extracting embeddings from {image_path}: {e}")
            return None
    
    def compare_embeddings(self, embedding1, embedding2):
        """Compare two face embeddings with enhanced accuracy"""
        if embedding1 is None or embedding2 is None:
            return 0.0, "INVALID_EMBEDDINGS"
        
        try:
            # Multi-metric comparison
            if 'insightface' in embedding1 and 'insightface' in embedding2:
                vec1 = embedding1['insightface']
                vec2 = embedding2['insightface']
            
                if isinstance(vec1, list):
                    vec1 = np.array(vec1)
                if isinstance(vec2, list):
                    vec2 = np.array(vec2)
                
                # Cosine similarity
                cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                # Euclidean distance
                euclidean_dist = np.linalg.norm(vec1 - vec2)
                euclidean_sim = 1 / (1 + euclidean_dist)
                
                # Weighted combination
                final_sim = 0.7 * cosine_sim + 0.3 * euclidean_sim
                
                # Enhanced confidence levels with stricter thresholds
                if final_sim > 0.80:
                    confidence = "EXCELLENT"
                elif final_sim > 0.70:
                    confidence = "VERY_HIGH"
                elif final_sim > 0.60:
                    confidence = "HIGH"
                elif final_sim > 0.50:
                    confidence = "MEDIUM"
                elif final_sim > 0.40:
                    confidence = "LOW"
                else:
                    confidence = "VERY_LOW"
                
                logger.info(f"🔍 Face comparison: {final_sim:.3f} similarity ({confidence})")
                return float(final_sim), confidence
            else:
                return 0.0, "NO_INSIGHTFACE_EMBEDDINGS"
                
        except Exception as e:
            logger.error(f"❌ Error comparing embeddings: {e}")
            return 0.0, "COMPARISON_ERROR"
    
    def validate_face_quality(self, embedding_data):
        """Validate if detected face meets quality standards"""
        if embedding_data is None:
            return False, "No embedding data"
        
        det_score = embedding_data.get('det_score', 0)
        
        # Enhanced quality thresholds based on algorithm
        algorithm = embedding_data.get('source', 'unknown')
        
        if algorithm == 'insightface':
            quality_threshold = 0.75
        elif algorithm == 'mtcnn':
            quality_threshold = 0.85
        elif algorithm == 'dlib':
            quality_threshold = 0.8
        else:
            quality_threshold = self.quality_threshold
        
        if det_score < quality_threshold:
            return False, f"Low detection score for {algorithm}: {det_score:.3f}"
        
        return True, f"Face quality acceptable ({algorithm}: {det_score:.3f})"
    
    def detect_and_match_faces_realtime(self, frame, lost_person_encodings, lost_person_names, threshold=0.65):
        """Real-time face detection and matching with immediate alerts"""
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Multi-algorithm face detection
            detected_faces = self.detect_faces_multi_algorithm(frame_rgb)
            
            matches = []
            current_time = datetime.now()
            
            for face in detected_faces:
                try:
                    # Extract embedding for this face
                    x1, y1, x2, y2 = map(int, face['bbox'])
                    # Clamp bbox to image bounds to avoid invalid crops
                    h, w = frame_rgb.shape[:2]
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))
                    if x2 <= x1 or y2 <= y1:
                        # Skip invalid bbox
                        continue
                    cropped_face = frame_rgb[y1:y2, x1:x2]
                    
                    if cropped_face.size > 0:
                        # Get embedding using InsightFace
                        insight_faces = self.insight_app.get(cropped_face)
                        
                        if len(insight_faces) > 0:
                            face_embedding = insight_faces[0].embedding
                            
                            # Compare with all lost persons
                            best_match = None
                            best_similarity = 0
                            best_confidence = ""
                            
                            for i, lost_embedding in enumerate(lost_person_encodings):
                                similarity, confidence = self.compare_embeddings(
                                    {'insightface': face_embedding}, 
                                    {'insightface': lost_embedding}
                                )
                                
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = lost_person_names[i]
                                    best_confidence = confidence
                            
                            # Check if match is above threshold
                            if best_similarity >= threshold and best_match:
                                match_info = {
                                    'name': best_match,
                                    'similarity': best_similarity,
                                    'confidence': best_confidence,
                                    'bbox': face['bbox'],
                                    'algorithm': face['algorithm'],
                                    'timestamp': current_time,
                                    'found': True
                                }
                                
                                matches.append(match_info)
                                
                                # Add to recent detections
                                self.recent_detections.append(match_info)
                                
                                # Add to found persons set
                                self.found_persons.add(best_match)
                                
                                logger.info(f"🎯 FOUND PERSON: {best_match} "
                                          f"(similarity: {best_similarity:.3f}, "
                                          f"confidence: {best_confidence}, "
                                          f"algorithm: {face['algorithm']})")
                                
                                # Keep only last 100 detections
                                if len(self.recent_detections) > 100:
                                    self.recent_detections = self.recent_detections[-100:]
                
                except Exception as e:
                    logger.error(f"❌ Error processing face: {e}")
                    continue
            
            return matches
            
        except Exception as e:
            logger.error(f"❌ Error in real-time face detection: {e}")
            return []
    
    def get_recent_detections(self, limit=10):
        """Get recent face detections with timestamps"""
        recent = self.recent_detections[-limit:]
        
        # Format for display
        formatted_detections = []
        for detection in recent:
            formatted_detections.append({
                'name': detection['name'],
                'similarity': f"{detection['similarity']:.3f}",
                'confidence': detection['confidence'],
                'timestamp': detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'time_ago': self.get_time_ago(detection['timestamp'])
            })
        
        return formatted_detections
    
    def get_time_ago(self, timestamp):
        """Get human-readable time ago"""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
    
    def clear_found_persons(self):
        """Clear the found persons cache"""
        self.found_persons.clear()
        logger.info("🧹 Cleared found persons cache")
    
    def get_found_persons_count(self):
        """Get count of unique found persons"""
        return len(self.found_persons)