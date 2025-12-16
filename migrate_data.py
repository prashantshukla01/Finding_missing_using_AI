import json
import os
from datetime import datetime
from flask import Flask
from config import Config
from models.db_models import db, Person, Stream, Detection

def parse_date(date_str):
    if not date_str:
        return None
    try:
        # Try full ISO format
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except ValueError:
        try:
            # Try simple date format
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None

def migrate_data():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Override SQLALCHEMY_DATABASE_URI
    db_path = os.path.join(app.config['DATABASE_PATH'], 'detection_system.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    json_path = app.config['DATABASE_PATH']
    
    with app.app_context():
        print("Starting migration...")
        
        # 1. Migrate Persons
        persons_file = os.path.join(json_path, 'persons.json')
        if os.path.exists(persons_file):
            with open(persons_file, 'r') as f:
                persons_data = json.load(f)
                
            count = 0
            for p_id, p_data in persons_data.items():
                if not Person.query.get(p_id):
                    # Handle embedding - stored as dictionary in JSON, we need it as JSON string
                    embedding_data = p_data.get('embedding')
                    
                    person = Person(
                        id=p_id,
                        name=p_data.get('name'),
                        display_name=p_data.get('display_name'),
                        age=p_data.get('age'),
                        description=p_data.get('description'),
                        last_seen_location=p_data.get('last_seen_location'),
                        last_seen_time=p_data.get('last_seen_time'),
                        contact_info=p_data.get('contact_info'),
                        additional_notes=p_data.get('additional_notes'),
                        image_path=p_data.get('image_path'),
                        embedding=embedding_data,
                        created_at=parse_date(p_data.get('created_at')) or datetime.utcnow()
                    )
                    db.session.add(person)
                    count += 1
            
            db.session.commit()
            print(f"Migrated {count} persons.")
            
        # 2. Migrate Streams
        streams_file = os.path.join(json_path, 'cctv_streams.json')
        stream_map = {} # Map name to ID for detections
        
        if os.path.exists(streams_file):
            with open(streams_file, 'r') as f:
                streams_data = json.load(f)
            
            count = 0
            for s_name, s_data in streams_data.items():
                existing = Stream.query.filter_by(name=s_name).first()
                if not existing:
                    stream = Stream(
                        name=s_name,
                        source_url=s_data.get('url'),
                        location=s_data.get('location'),
                        lat=float(s_data.get('lat')) if s_data.get('lat') else None,
                        lng=float(s_data.get('lng')) if s_data.get('lng') else None,
                        active=s_data.get('active', False),
                        added_date=parse_date(s_data.get('added_date')) or datetime.utcnow()
                    )
                    db.session.add(stream)
                    db.session.flush() # Get ID
                    stream_map[s_name] = stream.id
                    count += 1
                else:
                    stream_map[s_name] = existing.id
            
            db.session.commit()
            print(f"Migrated {count} streams.")

        # 3. Migrate Detections
        detections_file = os.path.join(json_path, 'detections.json')
        if os.path.exists(detections_file):
            with open(detections_file, 'r') as f:
                detections_data = json.load(f)
                
            count = 0
            for d_data in detections_data:
                d_id = d_data.get('id')
                if not d_id: continue
                
                if not Detection.query.get(d_id):
                    # Try to link to Person
                    # JSON stores person_name, which might be name or ID or display name
                    # In persons.json, keys are UUIDs. 
                    # Detections 'person_name' looks like "5f8a..._profile" which matches 'name' field in person
                    
                    person_ref = d_data.get('person_name')
                    person = Person.query.filter_by(name=person_ref).first()
                    
                    # Try to link to Stream
                    stream_ref = d_data.get('stream_name')
                    stream_id = stream_map.get(stream_ref)
                    
                    detection = Detection(
                        id=d_id,
                        person_id=person.id if person else None,
                        stream_id=stream_id,
                        person_name=d_data.get('person_name'),
                        stream_name=d_data.get('stream_name'),
                        timestamp=parse_date(d_data.get('timestamp')) or datetime.utcnow(),
                        confidence=d_data.get('similarity')
                    )
                    db.session.add(detection)
                    count += 1
            
            db.session.commit()
            print(f"Migrated {count} detections.")

if __name__ == '__main__':
    migrate_data()
