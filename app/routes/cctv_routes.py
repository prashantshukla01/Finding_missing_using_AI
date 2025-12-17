from flask import Blueprint, render_template, request, jsonify
import logging
from datetime import datetime
from app.models.db_models import db, Person

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
        
        # Get persons count from SQL
        try:
            persons_count = Person.query.count()
        except Exception:
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