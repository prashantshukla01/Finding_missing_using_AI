import cv2
import threading
import time
import logging
import json
import base64
import cv2
import numpy as np
import time
import os
from datetime import datetime
from queue import Queue

logger = logging.getLogger(__name__)

class CCTVManager:
    def __init__(self, config):
        self.config = config
        self.active_streams = {}
        self.frame_queues = {}
        self.stream_threads = {}
        self.running = False
        
        # Lost persons database
        self.lost_face_encodings = []
        self.lost_face_names = []
        self.lost_faces_dir = "data/lost_faces"
        
        # Load existing streams from database
        self.load_streams_from_db()
        
        # Load lost persons database
        self.load_lost_persons_database()
        
        logger.info("CCTV Manager initialized successfully")
    
    def load_lost_persons_database(self):
        """Load and encode all lost persons from the database directory using insightface"""
        try:
            if not os.path.exists(self.lost_faces_dir):
                os.makedirs(self.lost_faces_dir, exist_ok=True)
                logger.info(f"Created lost persons directory: {self.lost_faces_dir}")
                return
            
            self.lost_face_encodings = []
            self.lost_face_names = []
            
            image_files = [f for f in os.listdir(self.lost_faces_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            logger.info(f"Found {len(image_files)} images in lost persons database")
            
            # Check if face_matcher is available
            if not hasattr(self.config, 'face_matcher'):
                logger.warning("Face matcher not available yet, skipping face encoding for existing images")
                return
            
            for image_file in image_files:
                try:
                    image_path = os.path.join(self.lost_faces_dir, image_file)
                    
                    # Use insightface to extract embeddings directly
                    embedding_data = self.config.face_matcher.extract_embeddings(image_path)
                    
                    if embedding_data is not None:
                        # Validate face quality
                        is_valid, message = self.config.face_matcher.validate_face_quality(embedding_data)
                        
                        if is_valid:
                            # Extract name from filename (remove extension)
                            name = os.path.splitext(image_file)[0]
                            self.lost_face_encodings.append(embedding_data['insightface'])
                            self.lost_face_names.append(name)
                            logger.info(f"✅ Encoded lost person: {name}")
                        else:
                            logger.warning(f"⚠️ Face quality check failed for {image_file}: {message}")
                    else:
                        logger.warning(f"⚠️ No face found in {image_file}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {image_file}: {e}")
            
            logger.info(f"✅ Loaded {len(self.lost_face_names)} lost persons into database")
            
        except Exception as e:
            logger.error(f"❌ Error loading lost persons database: {e}")
    
    def reload_lost_persons_database(self):
        """Reload lost persons database after face_matcher is available"""
        try:
            if not hasattr(self.config, 'face_matcher'):
                logger.error("Cannot reload: face_matcher not available")
                return False
            
            logger.info("Reloading lost persons database with face_matcher...")
            
            # Clear existing encodings
            self.lost_face_encodings = []
            self.lost_face_names = []
            
            image_files = [f for f in os.listdir(self.lost_faces_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files:
                try:
                    image_path = os.path.join(self.lost_faces_dir, image_file)
                    
                    # Use insightface to extract embeddings directly
                    embedding_data = self.config.face_matcher.extract_embeddings(image_path)
                    
                    if embedding_data is not None:
                        # Validate face quality
                        is_valid, message = self.config.face_matcher.validate_face_quality(embedding_data)
                        
                        if is_valid:
                            # Extract name from filename (remove extension)
                            name = os.path.splitext(image_file)[0]
                            self.lost_face_encodings.append(embedding_data['insightface'])
                            self.lost_face_names.append(name)
                            logger.info(f"✅ Encoded lost person: {name}")
                        else:
                            logger.warning(f"⚠️ Face quality check failed for {image_file}: {message}")
                    else:
                        logger.warning(f"⚠️ No face found in {image_file}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {image_file}: {e}")
            
            logger.info(f"✅ Reloaded {len(self.lost_face_names)} lost persons into database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reloading lost persons database: {e}")
            return False
    
    def add_lost_person(self, image_path, person_name):
        """Add a new lost person to the database using insightface"""
        try:
            # Copy image to lost faces directory
            filename = f"{person_name}.jpg"
            destination = os.path.join(self.lost_faces_dir, filename)
            
            # Read and save the image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ Could not load image from {image_path}")
                return False
            
            cv2.imwrite(destination, image)
            
            # Use insightface to extract embeddings directly
            embedding_data = self.config.face_matcher.extract_embeddings(destination)
            
            if embedding_data is not None:
                # Validate face quality
                is_valid, message = self.config.face_matcher.validate_face_quality(embedding_data)
                
                if is_valid:
                    self.lost_face_encodings.append(embedding_data['insightface'])
                    self.lost_face_names.append(person_name)
                    
                    # Save to database
                    from utils.helpers import save_person_to_db
                    person_data = {
                        'name': person_name,
                        'image_path': destination,
                        'embedding': embedding_data,
                        'age': '',
                        'description': 'Added via API',
                        'last_seen_location': 'Unknown',
                        'last_seen_time': datetime.now().isoformat(),
                        'contact_info': '',
                        'additional_notes': ''
                    }
                    
                    if hasattr(self.config, 'PERSONS_DB_FILE'):
                        person_id = save_person_to_db(person_data, self.config.PERSONS_DB_FILE)
                        if person_id:
                            logger.info(f"✅ Added new lost person: {person_name} with ID: {person_id}")
                        else:
                            logger.warning(f"⚠️ Added {person_name} to memory but failed to save to database")
                    else:
                        logger.warning(f"⚠️ No PERSONS_DB_FILE configured, {person_name} added to memory only")
                    
                    return True
                else:
                    logger.warning(f"⚠️ Face quality check failed for {person_name}: {message}")
                    return False
            else:
                logger.warning(f"⚠️ No face found in uploaded image for {person_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error adding lost person {person_name}: {e}")
            return False
    
    def load_streams_from_db(self):
        """Load CCTV streams from database file"""
        try:
            if not hasattr(self.config, 'CCTV_DB_FILE'):
                logger.info("No CCTV_DB_FILE in config, starting fresh")
                return
                
            if not os.path.exists(self.config.CCTV_DB_FILE):
                logger.info("No CCTV database file found, starting fresh")
                return
                
            with open(self.config.CCTV_DB_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    logger.info("CCTV database file is empty")
                    return
                    
                streams_data = json.loads(content)
                    
            for stream_name, stream_info in streams_data.items():
                self.add_stream(
                    stream_name,
                    stream_info['url'],
                    stream_info['location'],
                    start_monitoring=False
                )
                
            logger.info(f"Loaded {len(streams_data)} streams from database")
        except Exception as e:
            logger.error(f"Error loading CCTV database: {e}")
    
    def save_streams_to_db(self):
        """Save CCTV streams to database file"""
        try:
            if not hasattr(self.config, 'CCTV_DB_FILE'):
                return
                
            streams_data = {}
            for stream_name, stream_info in self.active_streams.items():
                streams_data[stream_name] = {
                    'url': stream_info['url'],
                    'location': stream_info['location'],
                    'added_date': stream_info.get('added_date', datetime.now().isoformat())
                }
            
            with open(self.config.CCTV_DB_FILE, 'w') as f:
                json.dump(streams_data, f, indent=2)
                
            logger.info(f"Saved {len(streams_data)} streams to database")
        except Exception as e:
            logger.error(f"Error saving CCTV database: {e}")
    
    def add_stream(self, stream_name, rtsp_url, location, start_monitoring=True):
        """Add a new RTSP stream"""
        logger.info(f"Attempting to add stream: {stream_name}")
        
        if stream_name in self.active_streams:
            logger.warning(f"Stream {stream_name} already exists")
            return False
        
        # Test connection first (skip for webcam and demo)
        if rtsp_url not in ["0", "demo"] and not self.test_rtsp_connection(rtsp_url):
            logger.error(f"Failed to connect to RTSP stream: {rtsp_url}")
            return False
        
        self.active_streams[stream_name] = {
            'url': rtsp_url,
            'location': location,
            'active': True,
            'last_frame': None,
            'last_update': None,
            'added_date': datetime.now().isoformat(),
            'error_count': 0
        }
        
        # Create frame queue for this stream
        self.frame_queues[stream_name] = Queue(maxsize=1)
        
        if start_monitoring:
            self.start_stream_monitoring(stream_name)
        
        # Save to database
        self.save_streams_to_db()
        
        logger.info(f"Successfully added stream: {stream_name} at location: {location}")
        return True
    
    def add_webcam_stream(self, stream_name="Live Webcam", location="Your Location"):
        """Add webcam as a stream for testing"""
        logger.info(f"Attempting to add webcam stream: {stream_name}")

        webcam_url = "0"

        # ✅ Fix: Remove existing webcam stream if already present
        if stream_name in self.active_streams:
            logger.warning(f"Stream {stream_name} already exists — removing and reinitializing.")
            try:
                del self.active_streams[stream_name]
                logger.info(f"Removed old stream entry for {stream_name}.")
            except Exception as e:
                logger.error(f"Failed to remove old webcam stream: {e}")

        # Test if webcam is available
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    logger.info(f"Webcam is available, adding stream: {stream_name}")
                    return self.add_stream(stream_name, webcam_url, location)
                else:
                    logger.error("Webcam opened but cannot read frames")
            else:
                logger.error("Webcam not accessible or already in use.")
        except Exception as e:
            logger.error(f"Error testing webcam: {e}")

        logger.error("Failed to add webcam stream — switching to demo.")
        return False

    
    def test_rtsp_connection(self, rtsp_url):
        """Test if RTSP stream is accessible"""
        try:
            logger.info(f"Testing RTSP connection: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000)
            
            if not cap.isOpened():
                return False
            
            # Try to read a frame with timeout
            start_time = time.time()
            while (time.time() - start_time) < 10:
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return True
                time.sleep(0.03)
            
            cap.release()
            return False
            
        except Exception as e:
            logger.error(f"RTSP connection test failed for {rtsp_url}: {e}")
            return False
    
    def start_stream_monitoring(self, stream_name):
        """Start monitoring a specific stream"""
        if stream_name not in self.active_streams:
            logger.error(f"Stream {stream_name} not found")
            return False
        
        if stream_name in self.stream_threads and self.stream_threads[stream_name].is_alive():
            logger.warning(f"Stream {stream_name} is already being monitored")
            return True
        
        # Start monitoring thread
        self.running = True
        thread = threading.Thread(
            target=self._monitor_stream,
            args=(stream_name,),
            daemon=True
        )
        thread.start()
        self.stream_threads[stream_name] = thread
        
        logger.info(f"Started monitoring stream: {stream_name}")
        return True
    
    def _monitor_stream(self, stream_name):
        """Monitor stream and capture frames with webcam support"""
        stream_info = self.active_streams[stream_name]
        
        # Initialize webcam if this is a webcam stream
        cap = None
        if stream_info['url'] == "0":
            try:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 25)  # Set target FPS for smooth streaming
                logger.info(f"Webcam initialized for {stream_name}")
            except Exception as e:
                logger.error(f"Failed to initialize webcam: {e}")
                return
        
        while self.running and stream_info['active']:
            try:
                if stream_info['url'] == "0" and cap is not None:
                    # Read from webcam
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read from webcam")
                        time.sleep(0.05)
                        continue
                    # Resize for consistency
                    frame = cv2.resize(frame, (640, 480))
                else:
                    # Create demo frame for non-webcam streams
                    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                    cv2.putText(frame, f"Stream: {stream_name}", (50, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(frame, "Face Detection System", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(frame, "Look at webcam for face detection", (50, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (50, 300), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Update stream info
                stream_info['last_frame'] = frame
                stream_info['last_update'] = datetime.now()
                
                # Put frame in queue
                if not self.frame_queues[stream_name].empty():
                    try:
                        self.frame_queues[stream_name].get_nowait()
                    except:
                        pass
                
                self.frame_queues[stream_name].put(frame)
                
                time.sleep(0.1)  # 10 FPS for smooth video
                
            except Exception as e:
                logger.error(f"Error in stream monitoring for {stream_name}: {e}")
                time.sleep(0.05)
        
        # Release webcam when done
        if cap is not None:
            cap.release()
            logger.info(f"Webcam released for {stream_name}")
    
    def get_current_frame(self, stream_name):
        """Return the latest frame for a given stream, using queue fallback, placeholder,
        and lightweight face detection+matching overlays (red = matched, blue = unknown)."""
        try:
            if stream_name not in self.active_streams:
                logger.warning(f"Unknown stream requested: {stream_name}")
                return None

            frame = None
            # Try from queue (freshest)
            if stream_name in self.frame_queues and not self.frame_queues[stream_name].empty():
                try:
                    frame = self.frame_queues[stream_name].get_nowait()
                    self.active_streams[stream_name]["last_frame"] = frame
                except Exception:
                    frame = self.active_streams[stream_name].get("last_frame")
            else:
                # fallback to cached frame
                frame = self.active_streams[stream_name].get("last_frame")

            # Graceful startup placeholder if no frame yet
            if frame is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Initializing Webcam...", (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(placeholder, time.strftime("%H:%M:%S"), (240, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                ret, buffer = cv2.imencode(".jpg", placeholder)
                return buffer.tobytes()

            # --- Enhanced Real-time Face Detection & Matching with Advanced Algorithms ---
            try:
                # Throttle detection per-stream to ~0.1s for better performance
                now = time.time()
                last_det = self.active_streams[stream_name].get("_last_detect_time", 0)
                do_detect = (now - last_det) >= 0.1  # 10 FPS detection rate

                if do_detect and hasattr(self.config, "face_matcher") and self.config.face_matcher is not None:
                    # Use advanced multi-algorithm real-time detection
                    matches = self.config.face_matcher.detect_and_match_faces_realtime(
                        frame, 
                        self.lost_face_encodings, 
                        self.lost_face_names,
                        threshold=0.65
                    )
                    
                    overlays = []  # list of (x1,y1,x2,y2,name,conf, is_found)
                    
                    for match in matches:
                        x1, y1, x2, y2 = map(int, match['bbox'])
                        name = match['name']
                        confidence = match['similarity']
                        is_found = match['found']
                        
                        overlays.append((x1, y1, x2, y2, name, confidence, is_found))
                        
                        # Log found persons with enhanced information
                        if is_found:
                            logger.info(f"🚨 FOUND MISSING PERSON: {name} in stream {stream_name} "
                                      f"(similarity: {confidence:.3f}, "
                                      f"confidence: {match['confidence']}, "
                                      f"algorithm: {match['algorithm']})")
                    
                    # Store overlays and update last detect time
                    if overlays:
                        self.active_streams[stream_name]['_last_overlays'] = overlays
                        # Log found persons
                        for _, _, _, _, name, conf, is_found in overlays:
                            if is_found:
                                logger.info(f"[ALERT] {name} FOUND at {stream_name} with confidence {conf:.3f}")
                    else:
                        self.active_streams[stream_name].pop('_last_overlays', None)
                    self.active_streams[stream_name]['_last_detect_time'] = now

            except Exception as e:
                # Ensure detection errors never break frame serving
                logger.debug(f"Face detection/matching skipped due to error: {e}")

            # Draw overlays with FOUND/Unknown labeling
            overlays_to_draw = self.active_streams[stream_name].get('_last_overlays', [])
            found_persons = []  # Track found persons for alert banner
            
            try:
                for (x1, y1, x2, y2, name, conf, is_found) in overlays_to_draw:
                    # Clamp coordinates
                    h, w = frame.shape[:2]
                    x1c, y1c = max(0, int(x1)), max(0, int(y1))
                    x2c, y2c = min(w - 1, int(x2)), min(h - 1, int(y2))

                    # Choose color: RED for FOUND persons, BLUE for unknown
                    if is_found:
                        color = (0, 0, 255)   # Red (BGR) for FOUND persons
                        label = f"🚨 {name} FOUND"
                        found_persons.append(name)
                        # Draw thicker rectangle for found persons
                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 3)
                    else:
                        color = (255, 0, 0)   # Blue (BGR) for unknown
                        label = "Unknown"
                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), color, 2)

                    # Draw label with background for better visibility
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1c, y1c - th - 8), (x1c + tw + 4, y1c), color, -1)
                    cv2.putText(frame, label, (x1c + 2, y1c - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw alert banner at top if any persons were found
                if found_persons:
                    alert_text = f"🚨 FOUND: {', '.join(found_persons)} 🚨"
                    (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
                    banner_y = 50
                    # Draw background rectangle for banner
                    cv2.rectangle(frame, (10, banner_y - th - 10), (10 + tw + 20, banner_y + 10), (0, 0, 255), -1)
                    cv2.putText(frame, alert_text, (20, banner_y),
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
                    
            except Exception as e:
                logger.debug(f"Overlay drawing skipped: {e}")

            # --- End detection & overlay block ---
            if not overlays_to_draw:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                    if not os.path.exists(cascade_path):
                        logger.error(f"❌ Haarcascade not found at {cascade_path}")
                    face_cascade = cv2.CascadeClassifier(cascade_path)

                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # Show Unknown when no match overlays exist
                        cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                except Exception as e:
                    logger.warning(f"OpenCV fallback detection failed: {e}")

            # Encode real frame (with overlays)
            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                logger.error(f"Frame encoding failed for {stream_name}")
                return None

            return buffer.tobytes()

        except Exception as e:
            logger.error(f"Error retrieving current frame for {stream_name}: {e}")
            return None


    
    def get_stream_status(self):
        """Get status of all streams"""
        status = {}
        for stream_name, stream_info in self.active_streams.items():
            status[stream_name] = {
                'location': stream_info['location'],
                'active': stream_info['active'],
                'last_update': stream_info['last_update'],
                'error_count': stream_info['error_count'],
                'url': stream_info['url']
            }
        return status
    
    def stop_all_streams(self):
        """Stop all stream monitoring"""
        self.running = False
        for thread in self.stream_threads.values():
            thread.join(timeout=5)
        
        logger.info("All stream monitoring stopped")
        
        