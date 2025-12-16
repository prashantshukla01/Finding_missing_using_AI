import json
import os
import uuid
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

# JSON DB functions removed as part of SQLite migration

def allowed_file(filename, allowed_extensions=None):
    """Check if file extension is allowed"""
    if allowed_extensions is None:
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, upload_folder, subfolder=''):
    """Save uploaded file to server"""
    try:
        if not allowed_file(file.filename):
            return None, "File type not allowed"
        
        # Create filename
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(upload_folder, subfolder, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save file
        file.save(filepath)
        
        return filepath, None
        
    except Exception as e:
        return None, str(e)