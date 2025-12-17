from flask import Blueprint, render_template, request, jsonify
import logging
from app.models.db_models import db, SystemSettings

logger = logging.getLogger(__name__)

settings_bp = Blueprint('settings', __name__, url_prefix='/settings')

# Default Defaults
DEFAULT_SETTINGS = {
    'detection_threshold': '0.6',
    'unknown_alerts': 'false',
    'retention_days': '30',
    'debug_mode': 'false'
}

@settings_bp.route('/')
def settings_page():
    """Render settings UI"""
    return render_template('settings.html')

@settings_bp.route('/api/get', methods=['GET'])
def get_settings():
    """Get all system settings"""
    try:
        settings = {}
        # Fetch all from DB
        db_settings = SystemSettings.query.all()
        for s in db_settings:
            settings[s.key] = s.value
            
        # Merge with defaults for missing keys
        for key, default_val in DEFAULT_SETTINGS.items():
            if key not in settings:
                settings[key] = default_val
                
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        logger.error(f"Error fetching settings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@settings_bp.route('/api/save', methods=['POST'])
def save_settings():
    """Save system settings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        # Save each key
        for key, value in data.items():
            # Convert booleans/numbers to string for storage
            str_value = str(value).lower() if isinstance(value, bool) else str(value)
            
            SystemSettings.set_value(key, str_value)
            
        logger.info("System settings updated")
        return jsonify({'success': True, 'message': 'Settings saved successfully'})
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

def init_settings_routes(app):
    """Optional init hook"""
    pass
