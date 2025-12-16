import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask Config
    SECRET_KEY = os.getenv('SECRET_KEY', '4f6e8b3a9d2c7e1f5a8b3c6d9e2f4a7b1c8e3f6d9a2b5c8e1f4a7b0c3e6f9a2d5')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Face Recognition Config
    FACE_RECOGNITION_THRESHOLD = 0.5
    FACE_QUALITY_THRESHOLD = 0.7
    
    # CCTV Config
    RTSP_TIMEOUT = 10
    FRAME_CAPTURE_INTERVAL = 2  # seconds
    
    # Database Config
    DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'database')
    PERSONS_DB_FILE = os.path.join(DATABASE_PATH, 'persons.json')
    CCTV_DB_FILE = os.path.join(DATABASE_PATH, 'cctv_streams.json')
    DETECTIONS_DB_FILE = os.path.join(DATABASE_PATH, 'detections.json')
    
    # Model Config
    INSIGHTFACE_MODEL = 'buffalo_l'
    
    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(Config.UPLOAD_FOLDER, 'persons'), exist_ok=True)
        os.makedirs(os.path.join(Config.UPLOAD_FOLDER, 'temp'), exist_ok=True)
        os.makedirs(Config.DATABASE_PATH, exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}