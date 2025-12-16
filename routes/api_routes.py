from flask import Blueprint, request, jsonify
import logging
from datetime import datetime, timedelta
import json
import os
from utils.helpers import load_persons_from_db

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')

# MANUAL WORK: You need to initialize these in app.py
config = None
cctv_manager = None
face_matcher = None

def init_api_routes(app_config, cctv_manager_instance=None, face_matcher_instance=None):
    """Initialize API routes with application config and optional manager instances.

    This function should be called from `app.py` after creating the CCTV manager
    and face matcher so the routes can access those objects.
    """
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
        persons = load_persons_from_db(config.PERSONS_DB_FILE)
        
        # Remove embeddings from response to reduce payload size
        for person_id in persons:
            if 'embedding' in persons[person_id]:
                del persons[person_id]['embedding']
        
        return jsonify(persons)
    except Exception as e:
        logger.error(f"Error getting persons: {e}")
        return jsonify({}), 500

@api_bp.route('/detections/recent')
def get_recent_detections():
    """Get recent detections from advanced face matcher"""
    try:
        # Use the new face matcher to get recent detections
        # Read detections directly from database file
        if config and hasattr(config, 'DETECTIONS_DB_FILE') and os.path.exists(config.DETECTIONS_DB_FILE):
            with open(config.DETECTIONS_DB_FILE, 'r') as f:
                detections = json.load(f)
            
            # Sort by timestamp descending
            detections.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Load persons to get display names
            persons = load_persons_from_db(config.PERSONS_DB_FILE)
            
            # Map keys for frontend compatibility if needed
            result = []
            for d in detections[:50]:
                item = d.copy()
                if 'similarity' in d:
                    item['confidence'] = d['similarity']
                
                # Resolving Display Name
                # Resolving Display Name
                p_name = item.get('person_name')
                if p_name:
                    found_in_db = False
                    # 1. Try finding by filename match in persons DB
                    for p_id, p_data in persons.items():
                        p_img_path = p_data.get('image_path', '')
                        if not p_img_path: continue
                        
                        p_filename = os.path.basename(p_img_path)
                        p_stem = os.path.splitext(p_filename)[0]
                        
                        if p_stem == p_name:
                            item['person_name'] = p_data.get('display_name', p_data.get('name', p_name))
                            found_in_db = True
                            break
                            
                    if not found_in_db:
                        if p_name in persons and 'display_name' in persons[p_name]:
                            item['person_name'] = persons[p_name]['display_name']
                        elif '_' in p_name:
                            # Fallback: Try to strip UUID if strictly assuming UUID_Name format
                            parts = p_name.split('_', 1)
                            if len(parts) > 1:
                                item['person_name'] = parts[1]
                
                result.append(item)
                
            return jsonify(result)
        else:
            return jsonify([])
            
    except Exception as e:
        logger.error(f"Error getting recent detections: {e}")
        return jsonify([])

@api_bp.route('/detections/history')
def get_detection_history():
    """Get historical detections with filters"""
    try:
        if not config or not hasattr(config, 'DETECTIONS_DB_FILE') or not os.path.exists(config.DETECTIONS_DB_FILE):
             return jsonify([])

        with open(config.DETECTIONS_DB_FILE, 'r') as f:
            detections = json.load(f)

        # Filters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        person_name = request.args.get('person_name')
        stream_name = request.args.get('stream_name')

        filtered = detections

        if start_date:
            filtered = [d for d in filtered if d['timestamp'] >= start_date]
        if end_date:
            filtered = [d for d in filtered if d['timestamp'] <= end_date]
        if person_name:
            filtered = [d for d in filtered if person_name.lower() in d.get('person_name', '').lower()]
        if stream_name:
            filtered = [d for d in filtered if stream_name == d.get('stream_name')]

        # Sort by timestamp descending
        filtered.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Limit results to 1000 to avoid huge payloads
        return jsonify(filtered[:1000])

    except Exception as e:
        logger.error(f"Error getting detection history: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/search', methods=['POST'])
def search_person():
    """Search for a person across all CCTV streams"""
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
        
        # Extract embeddings from search image
        embedding = face_matcher.extract_embeddings(temp_image_path)
        
        if embedding is None:
            # Clean up temp file
            import os
            try:
                os.remove(temp_image_path)
            except:
                pass
            return jsonify({'success': False, 'error': 'No face detected in the image'}), 400
        
        # Search across all persons in database
        persons = load_persons_from_db(config.PERSONS_DB_FILE)
        matches = []
        
        for person_id, person_info in persons.items():
            if 'embedding' not in person_info:
                continue
            
            similarity, confidence = face_matcher.compare_embeddings(
                embedding, 
                person_info['embedding']
            )
            
            if similarity > config.FACE_RECOGNITION_THRESHOLD:
                matches.append({
                    'person_id': person_id,
                    'name': person_info.get('name', 'Unknown'),
                    'similarity': similarity,
                    'confidence': confidence,
                    'image_path': person_info.get('image_path'),
                    'last_seen': person_info.get('last_seen_location')
                })
        
        # Clean up temp file
        try:
            os.remove(temp_image_path)
        except:
            pass
        
        # Sort matches by similarity (highest first)
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
        # Safely load persons count
        total_persons = 0
        if config and hasattr(config, 'PERSONS_DB_FILE'):
            try:
                persons = load_persons_from_db(config.PERSONS_DB_FILE)
                total_persons = len(persons)
            except:
                total_persons = 0
        
        # Safely get stream info
        total_streams = 0
        active_streams = 0
        if cctv_manager:
            try:
                stream_status = cctv_manager.get_stream_status()
                total_streams = len(stream_status)
                active_streams = sum(1 for stream in stream_status.values() if stream.get('active', False))
            except:
                total_streams = 0
                active_streams = 0
        
        # Safely get detections count
        detections_today = 0
        if config and hasattr(config, 'DETECTIONS_DB_FILE'):
            try:
                if os.path.exists(config.DETECTIONS_DB_FILE):
                    with open(config.DETECTIONS_DB_FILE, 'r') as f:
                        detections = json.load(f)
                    cutoff_time = datetime.now() - timedelta(hours=24)
                    detections_today = len([d for d in detections 
                                          if datetime.fromisoformat(d['timestamp']) > cutoff_time])
            except:
                detections_today = 0
        
        # Determine system status
        status = 'Operational'
        if not cctv_manager or not cctv_manager.running:
            status = 'Stopped'
        elif not face_matcher:
             status = 'Degraded (No Face Matcher)'
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
        # Return basic stats even if there's an error
        return jsonify({
            'total_persons': 0,
            'total_streams': 0,
            'active_streams': 0,
            'detections_today': 0,
            'system_status': 'degraded',
            'error': str(e)
        })