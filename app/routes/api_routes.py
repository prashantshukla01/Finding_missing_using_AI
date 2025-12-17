from flask import Blueprint, request, jsonify
import logging
from datetime import datetime, timedelta
from app.models.db_models import db, Person, Detection, Stream
from app.utils.helpers import allowed_file

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')

# MANUAL WORK: You need to initialize these in app.py
config = None
cctv_manager = None
face_matcher = None

def init_api_routes(app_config, cctv_manager_instance=None, face_matcher_instance=None):
    global config, cctv_manager, face_matcher
    config = app_config
    cctv_manager = cctv_manager_instance
    face_matcher = face_matcher_instance

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Missing Person Detection System'
    })

@api_bp.route('/persons')
def get_persons():
    """Get all registered persons"""
    try:
        persons_list = Person.query.all()
        persons = {p.id: p.to_dict() for p in persons_list}
        
        # Remove embeddings from response to reduce payload size
        for p_data in persons.values():
            if 'embedding' in p_data:
                del p_data['embedding']
        
        return jsonify(persons)
    except Exception as e:
        logger.error(f"Error getting persons: {e}")
        return jsonify({}), 500

@api_bp.route('/detections/recent')
def get_recent_detections():
    """Get recent detections"""
    try:
        detections = Detection.query.order_by(Detection.timestamp.desc()).limit(50).all()
        result = [d.to_dict() for d in detections]
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting recent detections: {e}")
        return jsonify([])

@api_bp.route('/detections/history')
def get_detection_history():
    """Get historical detections with filters"""
    try:
        query = Detection.query

        # Filters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        person_name = request.args.get('person_name')
        stream_name = request.args.get('stream_name')

        if start_date:
            try:
                s_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                query = query.filter(Detection.timestamp >= s_date)
            except: pass
            
        if end_date:
            try:
                e_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query = query.filter(Detection.timestamp <= e_date)
            except: pass
            
        if person_name:
             query = query.filter(Detection.person_name.ilike(f"%{person_name}%"))
             
        if stream_name:
             query = query.filter(Detection.stream_name == stream_name)

        # Sort and limit
        detections = query.order_by(Detection.timestamp.desc()).limit(1000).all()
        
        result = [d.to_dict() for d in detections]
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting detection history: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/search', methods=['POST'])
def search_person():
    """Search for a person using an uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400
        
        # Save temporary image
        from utils.helpers import save_uploaded_file
        temp_image_path, error = save_uploaded_file(
            image_file, 
            config.UPLOAD_FOLDER, 
            'temp'
        )
        
        if error:
            return jsonify({'success': False, 'error': f'File upload failed: {error}'}), 400
        
        # Extract embeddings
        embedding = face_matcher.extract_embeddings(temp_image_path)
        
        # Cleanup
        try:
             os.remove(temp_image_path)
        except: pass
        
        if embedding is None:
            return jsonify({'success': False, 'error': 'No face detected'}), 400
        
        # Compare with all persons
        persons = Person.query.all()
        matches = []
        
        for person in persons:
            if not person.embedding:
                continue
            
            similarity, confidence = face_matcher.compare_embeddings(
                embedding, 
                person.embedding
            )
            
            if similarity > config.FACE_RECOGNITION_THRESHOLD:
                matches.append({
                    'person_id': person.id,
                    'name': person.name,
                    'similarity': similarity,
                    'confidence': confidence,
                    'image_path': person.image_path,
                    'last_seen': person.last_seen_location
                })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return jsonify({
            'success': True,
            'matches_found': len(matches),
            'matches': matches
        })
        
    except Exception as e:
        logger.error(f"Error in person search: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/stats')
def get_system_stats():
    """Get system statistics"""
    try:
        total_persons = Person.query.count()
        
        # Active streams
        total_streams = 0
        active_streams = 0
        if cctv_manager:
            try:
                stream_status = cctv_manager.get_stream_status()
                total_streams = len(stream_status)
                active_streams = sum(1 for s in stream_status.values() if s.get('active', False))
            except: pass
            
        # Detections today
        cutoff_time = datetime.now() - timedelta(hours=24)
        detections_today = Detection.query.filter(Detection.timestamp > cutoff_time).count()
        
        status = 'Operational'
        if not cctv_manager or not cctv_manager.running:
            status = 'Stopped'
        elif active_streams == 0 and total_streams > 0:
             status = 'Idle (No Active Streams)'
             
        return jsonify({
            'total_persons': total_persons,
            'total_streams': total_streams,
            'active_streams': active_streams,
            'detections_today': detections_today,
            'system_status': status,
            'face_recognition': 'active' if face_matcher else 'inactive',
            'cctv_monitoring': 'active' if cctv_manager else 'inactive'
        })
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({
            'total_persons': 0, 'total_streams': 0, 'detections_today': 0, 
            'error': str(e)
        })