from flask import Blueprint, render_template, request, jsonify, send_file, abort
import logging
import os
from datetime import datetime
from utils.helpers import save_uploaded_file, allowed_file
from models.db_models import db, Person

logger = logging.getLogger(__name__)

person_bp = Blueprint('person', __name__, url_prefix='/person')

# Global variables that will be initialized
face_matcher = None
app_config = None
cctv_manager = None

def init_person_routes(config, face_matcher_instance, cctv_manager_instance=None):
    """Initialize the person routes with configuration"""
    global app_config, face_matcher, cctv_manager
    app_config = config
    face_matcher = face_matcher_instance
    cctv_manager = cctv_manager_instance
    logger.info("Person routes initialized with config and dependencies")

@person_bp.route('/test')
def test_config():
    """Test if config is working"""
    if not app_config:
        return jsonify({'error': 'Config is None'}), 500
    
    config_info = {
        'config_exists': app_config is not None,
        'has_upload_folder': hasattr(app_config, 'UPLOAD_FOLDER'),
        'upload_folder': getattr(app_config, 'UPLOAD_FOLDER', 'MISSING'),
        'face_matcher_exists': face_matcher is not None
    }
    
    return jsonify(config_info)

@person_bp.route('/upload', methods=['GET'])
def upload_person_form():
    """Display person upload form"""
    return render_template('upload_person.html')

@person_bp.route('/upload', methods=['POST'])
def upload_person():
    """Handle person details upload"""
    try:
        # Check if config and face_matcher are initialized
        if not app_config:
            logger.error("Config not initialized in person routes")
            return jsonify({'success': False, 'error': 'System configuration error - Config not set'}), 500
        
        if not face_matcher:
            logger.error("Face matcher not initialized in person routes")
            return jsonify({'success': False, 'error': 'System configuration error - Face recognition not available'}), 500

        # Check if config has required attributes
        if not hasattr(app_config, 'UPLOAD_FOLDER'):
            logger.error("UPLOAD_FOLDER missing from config")
            return jsonify({'success': False, 'error': 'System configuration error - Upload folder not configured'}), 500

        # Extract form data
        name = request.form.get('name', '').strip()
        age = request.form.get('age', '').strip()
        last_seen_location = request.form.get('last_seen_location', '').strip()
        last_seen_time = request.form.get('last_seen_time', '').strip()
        description = request.form.get('description', '').strip()
        contact_info = request.form.get('contact_info', '').strip()
        additional_notes = request.form.get('additional_notes', '').strip()
        
        # Validate required fields
        if not name:
            return jsonify({'success': False, 'error': 'Name is required'}), 400
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Image is required'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400
        
        # Validate file type
        if not allowed_file(image_file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Use JPG, PNG, or JPEG'}), 400
        
        # Save uploaded image
        image_path, error = save_uploaded_file(
            image_file, 
            app_config.UPLOAD_FOLDER, 
            'persons'
        )
        
        if error:
            return jsonify({'success': False, 'error': f'File upload failed: {error}'}), 400
        
        # Extract face embeddings
        logger.info(f"Extracting embeddings for {name}")
        embedding = face_matcher.extract_embeddings(image_path)
        
        if embedding is None:
            # Clean up uploaded image
            try:
                os.remove(image_path)
            except:
                pass
            return jsonify({'success': False, 'error': 'Could not detect a face in the image. Please upload a clearer image with a visible face.'}), 400
        
        # Validate face quality
        is_quality_ok, quality_message = face_matcher.validate_face_quality(embedding)
        if not is_quality_ok:
            try:
                os.remove(image_path)
            except:
                pass
            return jsonify({'success': False, 'error': f'Face quality issue: {quality_message}'}), 400
        
        # Create and save new person
        import uuid
        person_id = str(uuid.uuid4())
        
        new_person = Person(
            id=person_id,
            name=name,
            display_name=name,
            age=age,
            description=description,
            last_seen_location=last_seen_location,
            last_seen_time=last_seen_time,
            contact_info=contact_info,
            additional_notes=additional_notes,
            image_path=image_path,
            embedding=embedding,
            created_at=datetime.utcnow()
        )
        
        db.session.add(new_person)
        db.session.commit()
        
        logger.info(f"Successfully registered person: {name} with ID: {person_id}")
        return jsonify({
            'success': True,
            'person_id': person_id,
            'message': f'Person {name} registered successfully and is now being monitored'
        })
            
    except Exception as e:
        logger.error(f"Error in person upload: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500

@person_bp.route('/list')
def list_persons():
    """Display list of all registered persons"""
    try:
        persons_list = Person.query.all()
        # Convert to dictionary keyed by ID for template compatibility
        persons = {p.id: p.to_dict() for p in persons_list}
        return render_template('person_list.html', persons=persons)
    except Exception as e:
        logger.error(f"Error loading persons list: {e}")
        return render_template('person_list.html', persons={})

@person_bp.route('/image/<person_id>')
def get_person_image(person_id):
    """Serve person image"""
    try:
        person = Person.query.get(person_id)
        
        if not person or not person.image_path:
            return abort(404)
        
        if not os.path.exists(person.image_path):
            return abort(404)
            
        return send_file(person.image_path)
    except Exception as e:
        logger.error(f"Error serving image for person {person_id}: {e}")
        return abort(404)

@person_bp.route('/delete/<person_id>', methods=['POST'])
def delete_person(person_id):
    """Delete a person from the database"""
    try:
        person = Person.query.get(person_id)
        
        if not person:
            return jsonify({'success': False, 'error': 'Person not found'}), 404
            
        # Get image path before deleting data
        image_path = person.image_path
        
        # Delete from DB
        db.session.delete(person)
        db.session.commit()
            
        # Try to delete the image file
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logger.info(f"Deleted image file: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to delete image file {image_path}: {e}")
                
        # Reload CCTV manager to update in-memory face database
        if cctv_manager:
            try:
                cctv_manager.reload_lost_persons_database()
                logger.info("Reloaded CCTV manager face database")
            except Exception as e:
                logger.error(f"Failed to reload CCTV manager database: {e}")
        
        logger.info(f"Successfully deleted person {person_id}")
        return jsonify({'success': True, 'message': 'Person deleted successfully'})
        
    except Exception as e:
        logger.error(f"Error deleting person {person_id}: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500