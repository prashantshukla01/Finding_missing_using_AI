from flask import Blueprint, render_template, request, jsonify
import logging
from datetime import datetime
import cv2
import os
from utils.helpers import save_detection_to_db, load_persons_from_db
import json
import numpy as np

logger = logging.getLogger(__name__)

cctv_bp = Blueprint('cctv', __name__, url_prefix='/cctv')

# MANUAL WORK: You need to initialize these in app.py
cctv_manager = None
config = None
face_matcher = None

def init_cctv_routes(app_config, cctv_manager_instance, face_matcher_instance):
    global config, cctv_manager, face_matcher
    config = app_config
    cctv_manager = cctv_manager_instance
    face_matcher = face_matcher_instance

@cctv_bp.route('/management')
def cctv_management():
    """Display CCTV management page"""
    return render_template('cctv_management.html')

@cctv_bp.route('/history')
def history_map_view():
    """Display historical detections and map view"""
    return render_template('historical_map.html')

@cctv_bp.route('/dashboard')
def dashboard():
    """Display main dashboard"""
    try:
        stream_status = cctv_manager.get_stream_status() if cctv_manager else {}
        
        # Safe way to load persons - handle missing config
        persons_count = 0
        if config and hasattr(config, 'PERSONS_DB_FILE'):
            persons = load_persons_from_db(config.PERSONS_DB_FILE)
            persons_count = len(persons)
        else:
            persons_count = 0
    
        return render_template('dashboard.html', 
                             streams=stream_status, 
                             persons_count=persons_count)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        # Return dashboard with empty data if there's an error
        return render_template('dashboard.html', 
                             streams={}, 
                             persons_count=0)

@cctv_bp.route('/add_stream', methods=['POST'])
def add_cctv_stream():
    """Add a new CCTV stream"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data or 'url' not in data or 'location' not in data:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        stream_name = data['name'].strip()
        rtsp_url = data['url'].strip()
        location = data['location'].strip()
        lat = data.get('lat')
        lng = data.get('lng')

        
        if not stream_name or not rtsp_url or not location:
            return jsonify({'success': False, 'error': 'All fields are required'}), 400
        
        # Add stream to CCTV manager
        success = cctv_manager.add_stream(stream_name, rtsp_url, location, lat, lng)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Stream {stream_name} added successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to add stream. Please check the RTSP URL.'
            }), 400
            
    except Exception as e:
        logger.error(f"Error adding CCTV stream: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@cctv_bp.route('/streams')
def get_streams():
    """Get all CCTV streams status"""
    try:
        stream_status = cctv_manager.get_stream_status() if cctv_manager else {}
        return jsonify(stream_status)
    except Exception as e:
        logger.error(f"Error getting streams: {e}")
        return jsonify({}), 500

@cctv_bp.route('/stream/<stream_name>/frame')
def get_stream_frame(stream_name):
    """Get current frame from CCTV stream with face detection"""
    try:
        frame_base64 = cctv_manager.get_current_frame(stream_name, as_base64=True)
        
        response_data = {
            'has_frame': frame_base64 is not None,
            'stream_name': stream_name,
            'timestamp': datetime.now().isoformat()
        }
        
        if frame_base64:
            response_data['frame'] = frame_base64
            # Detection is already handled by CCTVManager
            response_data['recent_detections'] = []
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error getting frame from {stream_name}: {e}")
        return jsonify({
            'has_frame': False, 
            'error': str(e),
            'stream_name': stream_name
        }), 500

@cctv_bp.route('/retry/<stream_name>', methods=['POST'])
def retry_stream(stream_name):
    """Retry connecting to a stream"""
    try:
        if not cctv_manager or stream_name not in cctv_manager.active_streams:
            return jsonify({'success': False, 'error': 'Stream not found'}), 404
        
        stream_info = cctv_manager.active_streams[stream_name]
        
        # Test connection
        if cctv_manager.test_rtsp_connection(stream_info['url']):
            stream_info['active'] = True
            stream_info['error_count'] = 0
            cctv_manager.start_stream_monitoring(stream_name)
            return jsonify({'success': True, 'message': 'Stream reconnected'})
        else:
            return jsonify({'success': False, 'error': 'Failed to reconnect to stream'})
            
    except Exception as e:
        logger.error(f"Error retrying stream {stream_name}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@cctv_bp.route('/toggle/<stream_name>', methods=['POST'])
def toggle_stream_status(stream_name):
    """Toggle a stream on/off"""
    try:
        data = request.get_json()
        active = data.get('active', True)
        logger.warning(f"🔌 TOGGLE REQUEST: {stream_name} -> {active}")
        
        if cctv_manager.set_stream_active(stream_name, active):
            status = "enabled" if active else "disabled"
            return jsonify({'success': True, 'message': f'Stream {stream_name} {status}'})
        else:
            return jsonify({'success': False, 'error': 'Stream not found'}), 404
            
    except Exception as e:
        logger.error(f"Error toggling stream {stream_name}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def process_frame_for_detection(stream_name):
    """Process frame to detect registered persons with visual feedback"""
    detections = []
    
    try:
        if not face_matcher or not config:
            return detections
            
        frame = cctv_manager.get_current_frame(stream_name, as_base64=False)
        if frame is None:
            return detections
        
        # Load registered persons
        persons = load_persons_from_db(config.PERSONS_DB_FILE)
        
        # Extract faces from current frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_matcher.insight_app.get(rgb_frame)
        
        # Draw face detection results on frame
        for face in faces:
            # Draw face bounding box
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Check against registered persons
            person_detected = False
            for person_id, person_info in persons.items():
                if 'embedding' not in person_info:
                    continue
                
                target_embedding = person_info['embedding']
                similarity, confidence = face_matcher.compare_embeddings(
                    target_embedding, 
                    {'insightface': face.embedding}
                )
                
                if similarity > config.FACE_RECOGNITION_THRESHOLD:
                    # Person matched!
                    detection = {
                        'person_id': person_id,
                        'person_name': person_info.get('name', 'Unknown'),
                        'confidence': similarity,
                        'location': stream_name,
                        'bbox': bbox.tolist(),
                        'stream_name': stream_name
                    }
                    
                    detections.append(detection)
                    
                    # Draw recognition info on frame
                    label = f"{person_info['name']} ({similarity*100:.1f}%)"
                    cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Save detection to database
                    save_detection_to_db(detection, config.DETECTIONS_DB_FILE)
                    
                    logger.info(f"Detection: {person_info['name']} at {stream_name} "
                               f"with {similarity*100:.1f}% confidence")
                    person_detected = True
                    break
            
            # If no person matched, show "Unknown Person"
            if not person_detected:
                cv2.putText(frame, "Unknown Person", (bbox[0], bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Update the frame in the stream with detection boxes
        if len(faces) > 0:
            # Convert back to BGR and update the frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not cctv_manager.frame_queues[stream_name].empty():
                try:
                    cctv_manager.frame_queues[stream_name].get_nowait()
                except:
                    pass
            cctv_manager.frame_queues[stream_name].put(frame_bgr)
        
        return detections
        
    except Exception as e:
        logger.error(f"Error in frame detection: {e}")
        return detections