import cv2
import numpy as np
import logging
import insightface
from insightface.app import FaceAnalysis
from utils.augmentations import get_augmentations

logger = logging.getLogger(__name__)

class AdvancedFaceMatcher:
    def __init__(self, model_name='buffalo_l', det_size=(640, 640)):
        self.model_name = model_name
        self.det_size = det_size
        self.similarity_threshold = 0.6
        self.quality_threshold = 0.7
        self.augmentations = get_augmentations()
        
        try:
            self.insight_app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
            self.insight_app.prepare(ctx_id=0, det_size=det_size)
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise
    
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
        """Extract face embeddings from image using InsightFace only"""
        try:
            image = self.preprocess_image(image_path)
            if image is None:
                return None
            
            # Get faces using InsightFace
            faces = self.insight_app.get(image)
            
            if len(faces) == 0:
                logger.warning(f"No faces detected in {image_path}")
                return None
            
            # Get the best face (highest detection score)
            best_face = max(faces, key=lambda x: x.det_score)
            
            embedding_data = {
                'insightface': best_face.embedding,
                'det_score': best_face.det_score,
                'bbox': best_face.bbox.tolist() if hasattr(best_face.bbox, 'tolist') else best_face.bbox,
                'source': 'insightface'
            }
            
            # Add landmarks if available
            if hasattr(best_face, 'kps'):
                embedding_data['landmarks'] = best_face.kps.tolist()
            
            logger.info(f"Successfully extracted embeddings from {image_path}")
            return embedding_data
            
        except Exception as e:
            logger.error(f"Error extracting embeddings from {image_path}: {e}")
            return None
    
    def compare_embeddings(self, embedding1, embedding2):
        """Compare two face embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0, "INVALID_EMBEDDINGS"
        
        try:
            # InsightFace comparison only
            if 'insightface' in embedding1 and 'insightface' in embedding2:
                vec1 = embedding1['insightface']
                vec2 = embedding2['insightface']
            
                if isinstance(vec1, list):
                    vec1 = np.array(vec1)
                if isinstance(vec2, list):
                    vec2 = np.array(vec2)
                
                cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                # Determine confidence level
                if cosine_sim > 0.75:
                    confidence = "VERY_HIGH"
                elif cosine_sim > 0.65:
                    confidence = "HIGH"
                elif cosine_sim > 0.55:
                    confidence = "MEDIUM"
                elif cosine_sim > 0.45:
                    confidence = "LOW"
                else:
                    confidence = "VERY_LOW"
                
                return float(cosine_sim), confidence
            else:
                return 0.0, "NO_INSIGHTFACE_EMBEDDINGS"
                
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0, "COMPARISON_ERROR"
    
    def validate_face_quality(self, embedding_data):
        """Validate if detected face meets quality standards"""
        if embedding_data is None:
            return False, "No embedding data"
        
        if embedding_data.get('det_score', 0) < self.quality_threshold:
            return False, f"Low detection score: {embedding_data['det_score']}"
        
        return True, "Face quality acceptable"