from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import uuid
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

db = SQLAlchemy()

class Person(db.Model):
    __tablename__ = 'persons'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    display_name = db.Column(db.String(100))
    age = db.Column(db.String(20))
    description = db.Column(db.Text)
    last_seen_location = db.Column(db.String(200))
    last_seen_time = db.Column(db.String(50)) # Storing as ISO string to match JSON format
    contact_info = db.Column(db.String(200))
    additional_notes = db.Column(db.Text)
    image_path = db.Column(db.String(500))
    embedding_json = db.Column(db.Text) # Store embeddings as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    detections = db.relationship('Detection', backref='person', lazy=True)

    @property
    def embedding(self):
        return json.loads(self.embedding_json) if self.embedding_json else None

    @embedding.setter
    def embedding(self, value):
        self.embedding_json = json.dumps(value, cls=NumpyEncoder)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'age': self.age,
            'description': self.description,
            'last_seen_location': self.last_seen_location,
            'last_seen_time': self.last_seen_time,
            'contact_info': self.contact_info,
            'additional_notes': self.additional_notes,
            'image_path': self.image_path,
            'embedding': self.embedding,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Stream(db.Model):
    __tablename__ = 'streams'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    source_url = db.Column(db.String(500), nullable=False)
    location = db.Column(db.String(200))
    lat = db.Column(db.Float)
    lng = db.Column(db.Float)
    active = db.Column(db.Boolean, default=False)
    added_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    detections = db.relationship('Detection', backref='stream', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'url': self.source_url,
            'location': self.location,
            'lat': self.lat,
            'lng': self.lng,
            'active': self.active,
            'added_date': self.added_date.isoformat() if self.added_date else None
        }

class Detection(db.Model):
    __tablename__ = 'detections'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    person_id = db.Column(db.String(36), db.ForeignKey('persons.id'), nullable=True) # Allow null if person deleted? or cascade?
    stream_id = db.Column(db.Integer, db.ForeignKey('streams.id'), nullable=True) # Allow null if stream deleted?
    
    # Redundant fields for historical record even if relations are deleted
    person_name = db.Column(db.String(100)) 
    stream_name = db.Column(db.String(100))
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    confidence = db.Column(db.Float)
    
    def to_dict(self):
        return {
            'id': self.id,
            'person_name': self.person_name,
            'stream_name': self.stream_name,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'confidence': self.confidence,
            'location': self.stream.location if self.stream else "Unknown"
        }

class SystemSettings(db.Model):
    __tablename__ = 'system_settings'
    
    key = db.Column(db.String(50), primary_key=True)
    value = db.Column(db.String(500)) # Store values as strings, convert on usage
    description = db.Column(db.String(200))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @staticmethod
    def get_value(key, default=None):
        setting = SystemSettings.query.get(key)
        return setting.value if setting else default

    @staticmethod
    def set_value(key, value, description=None):
        setting = SystemSettings.query.get(key)
        if not setting:
            setting = SystemSettings(key=key, value=str(value), description=description)
            db.session.add(setting)
        else:
            setting.value = str(value)
            if description:
                setting.description = description
        db.session.commit()
        return setting
