import json
import os
import uuid
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def save_person_to_db(person_data, db_file):
    """Save person data to JSON database with NumPy support"""
    try:
        # Load existing data
        try:
            with open(db_file, 'r') as f:
                persons = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            persons = {}
        
        # Generate unique ID
        person_id = str(uuid.uuid4())
        person_data['id'] = person_id
        person_data['created_at'] = datetime.now().isoformat()
        
        # Convert NumPy arrays to lists for JSON serialization
        if 'embedding' in person_data and person_data['embedding'] is not None:
            embedding_data = person_data['embedding']
            if 'insightface' in embedding_data and isinstance(embedding_data['insightface'], np.ndarray):
                embedding_data['insightface'] = embedding_data['insightface'].tolist()
        
        # Save to database with custom encoder
        persons[person_id] = person_data
        
        with open(db_file, 'w') as f:
            json.dump(persons, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved person {person_data['name']} to database")
        return person_id
        
    except Exception as e:
        logger.error(f"Error saving person to database: {e}")
        return None

def load_persons_from_db(db_file):
    """Load all persons from database"""
    try:
        if not os.path.exists(db_file):
            return {}
            
        with open(db_file, 'r') as f:
            persons = json.load(f)
            
        # Convert embedding lists back to NumPy arrays when loading
        for person_id, person_data in persons.items():
            if 'embedding' in person_data and person_data['embedding'] is not None:
                embedding_data = person_data['embedding']
                if 'insightface' in embedding_data and isinstance(embedding_data['insightface'], list):
                    embedding_data['insightface'] = np.array(embedding_data['insightface'])
                    
        return persons
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {db_file}: {e}")
        # Try to fix corrupted file by creating a new one
        try:
            with open(db_file, 'w') as f:
                json.dump({}, f)
            return {}
        except:
            return {}
    except Exception as e:
        logger.error(f"Error loading persons from database: {e}")
        return {}

def save_detection_to_db(detection_data, db_file):
    """Save detection to database"""
    try:
        # Load existing detections
        try:
            with open(db_file, 'r') as f:
                detections = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            detections = []
        
        # Add new detection
        detection_data['id'] = str(uuid.uuid4())
        detection_data['timestamp'] = datetime.now().isoformat()
        detections.append(detection_data)
        
        # Keep only last 1000 detections to prevent file from growing too large
        if len(detections) > 1000:
            detections = detections[-1000:]
        
        with open(db_file, 'w') as f:
            json.dump(detections, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved detection for {detection_data.get('person_name', 'Unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving detection to database: {e}")
        return False

def allowed_file(filename, allowed_extensions=None):
    """Check if file extension is allowed"""
    if allowed_extensions is None:
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, upload_folder, subfolder=''):
    """Save uploaded file to server"""
    try:
        if not allowed_file(file.filename):
            return None, "File type not allowed"
        
        # Create filename
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(upload_folder, subfolder, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save file
        file.save(filepath)
        
        return filepath, None
        
    except Exception as e:
        return None, str(e)