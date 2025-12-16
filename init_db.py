from flask import Flask
from config import Config
from models.db_models import db
import os

def init_database():
    app = Flask(__name__)
    
    # Ensure config loads correctly
    app.config.from_object(Config)
    
    # Override SQLALCHEMY_DATABASE_URI to use SQLite
    db_path = os.path.join(app.config['DATABASE_PATH'], 'detection_system.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        # Create database directory if it doesn't exist
        os.makedirs(app.config['DATABASE_PATH'], exist_ok=True)
        
        print(f"Creating database at: {db_path}")
        db.create_all()
        print("Database tables created successfully!")

if __name__ == '__main__':
    init_database()
