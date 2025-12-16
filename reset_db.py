import os
import json
import shutil
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATABASE_DIR = os.path.join(DATA_DIR, 'database')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
PERSONS_DIR = os.path.join(UPLOADS_DIR, 'persons')
PERSONS_DB = os.path.join(DATABASE_DIR, 'persons.json')
DETECTIONS_DB = os.path.join(DATABASE_DIR, 'detections.json')

def reset_db():
    print("Resetting database...")
    
    # 1. Clear Persons Directory
    if os.path.exists(PERSONS_DIR):
        print(f"Clearing {PERSONS_DIR}...")
        shutil.rmtree(PERSONS_DIR)
        os.makedirs(PERSONS_DIR, exist_ok=True)
    
    # 2. Reset JSONs
    print(f"Resetting {PERSONS_DB}...")
    with open(PERSONS_DB, 'w') as f:
        json.dump({}, f)
        
    print(f"Resetting {DETECTIONS_DB}...")
    with open(DETECTIONS_DB, 'w') as f:
        json.dump([], f)
        
    print("Database cleared.")

if __name__ == "__main__":
    reset_db()
