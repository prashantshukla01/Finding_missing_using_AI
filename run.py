from app import create_app
import os
import cv2
import logging

app = create_app(os.getenv('FLASK_CONFIG') or 'default')
logger = logging.getLogger(__name__)

# Webcam initialization helper (extracted from original app.py)
def add_webcam_for_testing(app):
    """Add webcam stream for face detection testing"""
    try:
        cctv_manager = app.extensions.get('cctv_manager')
        if not cctv_manager:
            return False
            
        # Test if webcam is available
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                logger.info("Webcam is available, adding webcam stream")
                success = cctv_manager.add_webcam_stream(
                    "Live Webcam", 
                    "Connaught Place, Delhi", 
                    lat=28.6315, 
                    lng=77.2167, 
                    active=False
                )
                if success:
                    logger.info("Webcam stream added successfully")
                    return True
                else:
                    logger.error("Failed to add webcam stream")
            else:
                logger.warning("Webcam opened but cannot read frames")
        else:
            logger.warning("Webcam not available")
    except Exception as e:
        logger.error(f"Error testing webcam: {e}")
    
    return False

if __name__ == '__main__':
    with app.app_context():
        webcam_added = add_webcam_for_testing(app)
        if not webcam_added:
            logger.info("Using demo stream instead of webcam")
            try:
                cctv_manager = app.extensions['cctv_manager']
                cctv_manager.add_stream("Demo Stream", "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8", "Test Location")
            except:
                pass

    print("\n" + "="*50)
    print("MISSING PERSON DETECTION SYSTEM")
    print("="*50)
    print("Access the application at: http://localhost:8001")
    print("Available pages:")
    print("  - Dashboard: /")
    print("  - Add Person: /person/upload")
    print("  - CCTV Management: /cctv/management")
    print("="*50 + "\n")
    
    app.run(
        host='0.0.0.0', 
        port=8001, 
        debug=True,
        threaded=True
    )
