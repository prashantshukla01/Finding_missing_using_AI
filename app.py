from flask import Flask, render_template, jsonify, Response, abort, request
import logging
import time
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from config import config
from models.face_matcher import AdvancedFaceMatcher
from models.cctv_manager import CCTVManager

# Import routes
from routes.person_routes import person_bp, init_person_routes
from routes.cctv_routes import cctv_bp, init_cctv_routes
from routes.api_routes import api_bp, init_api_routes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_app(config_name='default'):
    """Application factory pattern"""
    app = Flask(__name__)
    app_config = config[config_name]()
    app.config.from_object(app_config)
    app_config.init_app(app)
    
    # Initialize Database
    from models.db_models import db
    db.init_app(app)
    
    # Initialize components
    try:
        with app.app_context():
            # MANUAL WORK: These might take time to download models on first run
            logger.info("Initializing Face Matcher...")
            face_matcher = AdvancedFaceMatcher()
            
            logger.info("Initializing CCTV Manager...")
            cctv_manager = CCTVManager(app_config, app=app)
            app_config.face_matcher = face_matcher
            
            # Reload lost persons database now that face_matcher is available
            cctv_manager.reload_lost_persons_database()

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    # Initialize routes with dependencies - PASS app_config NOT app.config
    init_person_routes(app_config, face_matcher, cctv_manager)
    init_cctv_routes(app_config, cctv_manager, face_matcher)
    init_api_routes(app_config, cctv_manager, face_matcher)
    
    # Register blueprints
    app.register_blueprint(person_bp)
    app.register_blueprint(cctv_bp)
    app.register_blueprint(api_bp)
    
    def add_test_streams():
        """Add public test streams for demonstration"""
        test_streams = [
            {
                "name": "Test Stream 1", 
                "url": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
                "location": "Public Test"
            },
            {
                "name": "Test Stream 2",
                "url": "rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov", 
                "location": "Public Test"
            }
        ]
        
        for stream in test_streams:
            try:
                cctv_manager.add_stream(stream["name"], stream["url"], stream["location"])
                logger.info(f"Added test stream: {stream['name']}")
            except Exception as e:
                logger.warning(f"Failed to add test stream {stream['name']}: {e}")

    # Add test streams - commented out to speed up startup for testing
    # add_test_streams()
    
    def add_webcam_for_testing():
        """Add webcam stream for face detection testing"""
        try:
            # Test if webcam is available
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    logger.info("Webcam is available, adding webcam stream")
                    # Added active=False to keeping it OFF by default as requested
                    # Default to Connaught Place, New Delhi for map visualization
                    success = cctv_manager.add_webcam_stream(
                        "Live Webcam", 
                        "Connaught Place, Delhi", 
                        lat=28.6315, 
                        lng=77.2167, 
                        active=False
                    )
                    if success:
                        logger.info("Webcam stream added successfully")
                        return True
                    else:
                        logger.error("Failed to add webcam stream")
                else:
                    logger.warning("Webcam opened but cannot read frames")
            else:
                logger.warning("Webcam not available")
        except Exception as e:
            logger.error(f"Error testing webcam: {e}")
        
        return False

    # Add webcam stream
    with app.app_context():
        webcam_added = add_webcam_for_testing()

        if not webcam_added:
            logger.info("Using demo stream instead of webcam")
            # Ensure demo stream exists
            try:
                cctv_manager.add_stream("Demo Stream", "demo", "Test Location")
            except:
                pass
    @app.route('/api/cctv/stream/<name>/frame')
    def get_stream_frame(name):
        try:
            frame = cctv_manager.get_current_frame(name)
            if frame is None:
                abort(404)
            return Response(frame, mimetype='image/jpeg')
        except Exception as e:
            logger.error(f"Error serving frame for {name}: {e}")
            abort(500)
    
    @app.route('/api/cctv/stream/<name>')
    def video_feed(name):
        """Continuous MJPEG streaming for real-time video"""
        def generate():
            while True:
                try:
                    frame = cctv_manager.get_current_frame(name)
                    if frame:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(0.04)  # ~25 FPS for smooth streaming
                except Exception as e:
                    logger.error(f"Error generating MJPEG stream for {name}: {e}")
                    # Return a placeholder frame on error
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Stream Error", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(1)  # Wait before retrying
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/api/lost-person/add', methods=['POST'])
    def add_lost_person():
        """API endpoint to add a new lost person to the database"""
        try:
            logger.info("Received request to add lost person")
            
            if 'image' not in request.files:
                logger.error("No image file in request")
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                logger.error("Empty filename")
                return jsonify({'error': 'No file selected'}), 400
            
            name = request.form.get('name', '').strip()
            if not name:
                logger.error("No name provided")
                return jsonify({'error': 'Person name is required'}), 400
            
            logger.info(f"Adding lost person: {name}")
            
            # Generate Unique ID
            import uuid
            unique_id = f"{uuid.uuid4()}"
            safe_name = secure_filename(name)
            
            # Use UUID for filename/internal ID
            filename = secure_filename(f"{unique_id}_{safe_name}.jpg")
            logger.info(f"Generated filename with UUID: {filename}")
            
            # Save the file temporarily
            temp_path = os.path.join('data', 'uploads', 'temp', filename)
            logger.info(f"Temp path: {temp_path}")
            
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            file.save(temp_path)
            logger.info(f"File saved to: {temp_path}")
            
            # Add to lost persons database
            # Pass unique_id+name as the ID, and original name as display name
            person_id_name = f"{unique_id}_{safe_name}"
            success = cctv_manager.add_lost_person(temp_path, person_id_name, display_name=name)
            logger.info(f"add_lost_person returned: {success}")
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Lost person {name} added successfully',
                    'name': name,
                    'filename': filename
                })
            else:
                return jsonify({'error': 'Failed to process face image. Please ensure the image contains a clear face.'}), 400
                
        except Exception as e:
            logger.error(f"Error adding lost person: {e}", exc_info=True)
            return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# Root route
    
    
    @app.route('/')
    def index():
        return render_template('dashboard.html')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return """
        <html>
            <head><title>404 - Page Not Found</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
                <h1>404 - Page Not Found</h1>
                <p>The page you are looking for doesn't exist.</p>
                <a href="/">Go to Dashboard</a>
            </body>
        </html>
        """, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    logger.info("Missing Person Detection System initialized successfully")
    return app

if __name__ == '__main__':
    # MANUAL WORK: 
    # 1. First run might download AI models (can take several minutes)
    # 2. Make sure you have proper RTSP URLs for CCTV cameras
    # 3. Ensure all directories are created properly
    
    app = create_app()
    
    print("\n" + "="*50)
    print("MISSING PERSON DETECTION SYSTEM")
    print("="*50)
    print("Access the application at: http://localhost:8001")
    print("Available pages:")
    print("  - Dashboard: /")
    print("  - Add Person: /person/upload")
    print("  - CCTV Management: /cctv/management")
    print("="*50 + "\n")
    
    # Run the app
    app.run(
        host='0.0.0.0', 
        port=8001, 
        debug=True,
        threaded=True  # Important for handling multiple CCTV streams
    )