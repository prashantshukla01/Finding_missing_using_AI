from flask import Flask, render_template, jsonify
import logging
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
    app_config.init_app(app)
    
    # Initialize components
    try:
        # MANUAL WORK: These might take time to download models on first run
        logger.info("Initializing Face Matcher...")
        face_matcher = AdvancedFaceMatcher()
        
        logger.info("Initializing CCTV Manager...")
        cctv_manager = CCTVManager(app_config)
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    # Initialize routes with dependencies - PASS app_config NOT app.config
    init_person_routes(app_config, face_matcher)
    init_cctv_routes(app_config, cctv_manager, face_matcher)
    init_api_routes(app_config, cctv_manager)
    
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

    # Add test streams
    add_test_streams()
    
    # Add a simple demo stream
    try:
        success = cctv_manager.add_stream("Demo Stream", "demo", "Test Location")
        if success:
            logger.info("Demo stream added successfully")
        else:
            logger.error("Failed to add demo stream")
    except Exception as e:
        logger.error(f"Error adding demo stream: {e}")

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