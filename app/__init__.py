from flask import Flask, render_template
import logging
from config import config

# Import logic for init
# Note: models and routes are imported inside create_app or init functions to avoid circular deps if needed

def create_app(config_name='default'):
    """Application factory pattern"""
    app = Flask(__name__)
    app_config = config[config_name]()
    app.config.from_object(app_config)
    app_config.init_app(app)
    
    # Initialize Database
    from app.models.db_models import db
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
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
    
    # Initialize components
    try:
        with app.app_context():
            from app.models.face_matcher import AdvancedFaceMatcher
            from app.models.cctv_manager import CCTVManager
            
            logger.info("Initializing Face Matcher...")
            face_matcher = AdvancedFaceMatcher()
            
            logger.info("Initializing CCTV Manager...")
            cctv_manager = CCTVManager(app_config, app=app)
            app_config.face_matcher = face_matcher
            
            # Reload lost persons database
            cctv_manager.reload_lost_persons_database()
            
            # Store in app extensions for easy access
            app.extensions['cctv_manager'] = cctv_manager
            app.extensions['face_matcher'] = face_matcher

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
    
    # Initialize routes with dependencies
    # We will import the init functions from the blueprints
    from app.routes.person_routes import init_person_routes, person_bp
    from app.routes.cctv_routes import init_cctv_routes, cctv_bp
    from app.routes.api_routes import init_api_routes, api_bp
    from app.routes.api_routes import init_api_routes, api_bp
    from app.routes.main_routes import main_bp
    from app.routes.settings_routes import settings_bp
    
    init_person_routes(app_config, face_matcher, cctv_manager)
    init_cctv_routes(app_config, cctv_manager, face_matcher)
    init_api_routes(app_config, cctv_manager, face_matcher)
    
    # Register blueprints
    app.register_blueprint(person_bp)
    app.register_blueprint(cctv_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(settings_bp)

    # Setup test/demo streams (logic moved here or into a utils function)
    # For now, we keep the simple structure
    with app.app_context():
        # Add webcam logic
        # For simplicity, we can move this to a separate init/utils file or keep succinct here
        pass # The original logic was complex, valid to move to a helper

    @app.errorhandler(404)
    def not_found(error):
        return render_template('404.html'), 404 # Assuming 404.html exists or inline

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    logger.info("Missing Person Detection System initialized successfully")
    return app
