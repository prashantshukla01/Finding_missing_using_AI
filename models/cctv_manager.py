import cv2
import threading
import time
import logging
import json
import base64
import numpy as np
import os
from datetime import datetime
from queue import Queue, Empty
from collections import deque

logger = logging.getLogger(__name__)

class StreamReader:
    """
    Dedicated thread for reading frames as fast as possible.
    Maintains a buffer of size 1 to ensure we always get the LATEST frame.
    """
    def __init__(self, url):
        self.url = url
        # Handle webcam index
        if str(url).isdigit():
            self.cap = cv2.VideoCapture(int(url))
        else:
            self.cap = cv2.VideoCapture(url)
            
        self.q = deque(maxlen=1)
        self.running = True
        self.status = 'connecting'
        self.lock = threading.Lock()
        
        # Start reading thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.q.append(frame)
                    with self.lock: self.status = 'online'
                else:
                    with self.lock: self.status = 'error'
                    # Try to reconnect? For now just wait a bit to avoid spin loop
                    time.sleep(0.1)
            else:
                with self.lock: self.status = 'error'
                time.sleep(1)
                
    def read(self):
        try:
            return self.q.pop()
        except IndexError:
            return None
            
    def release(self):
        self.running = False
        # Try to join thread
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Force garbage collection to ensure hardware release on some platforms
        import gc
        gc.collect()
        
    def isOpened(self):
        return self.cap.isOpened()
        
    def reconnect(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.url)

class CCTVManager:
    def __init__(self, config):
        self.config = config
        self.active_streams = {}
        # Stores the latest frame for each stream: {stream_name: frame}
        self.latest_frames = {}
        # Lock for thread-safe access to active_streams and latest_frames
        self.lock = threading.Lock()
        
        self.stream_threads = {}
        self.running = True  # Global running flag
        self.latest_detections = {} # Cache for background detections
        self.last_detection_times = {} # Check for duplicate detections: {(stream_name, person_name): timestamp}
        
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
            
            with self.lock:
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
        self.load_lost_persons_database()
    
    def add_lost_person(self, image_path, person_name, display_name=None):
        """Add a new lost person to the database using insightface"""
        try:
            if display_name is None:
                display_name = person_name

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
                    with self.lock:
                        self.lost_face_encodings.append(embedding_data['insightface'])
                        self.lost_face_names.append(person_name)
                    
                    # Save to database
                    from utils.helpers import save_person_to_db
                    person_data = {
                        'name': person_name, # This is the ID/Filename
                        'display_name': display_name, # Human readable name
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
                return
                
            if not os.path.exists(self.config.CCTV_DB_FILE):
                return
                
            with open(self.config.CCTV_DB_FILE, 'r') as f:
                content = f.read().strip()
                if not content:
                    return
                streams_data = json.loads(content)
                    
            for stream_name, stream_info in streams_data.items():
                # Force all streams to start as INACTIVE/OFFLINE by default as requested
                self.add_stream(
                    stream_name,
                    stream_info['url'],
                    stream_info['location'],
                    lat=stream_info.get('lat'),
                    lng=stream_info.get('lng'),
                    start_monitoring=False # User requested all toggles off by default
                )
                
            logger.info(f"Loaded {len(streams_data)} streams from database (Started as INACTIVE)")
        except Exception as e:
            logger.error(f"Error loading CCTV database: {e}")
    
    def save_streams_to_db(self):
        """Save CCTV streams to database file"""
        try:
            if not hasattr(self.config, 'CCTV_DB_FILE'):
                return
                
            streams_data = {}
            with self.lock:
                for stream_name, stream_info in self.active_streams.items():
                    streams_data[stream_name] = {
                        'url': stream_info['url'],
                        'location': stream_info['location'],
                        'lat': stream_info.get('lat'),
                        'lng': stream_info.get('lng'),
                        'active': stream_info.get('active', True),
                        'added_date': stream_info.get('added_date', datetime.now().isoformat())
                    }
            
            with open(self.config.CCTV_DB_FILE, 'w') as f:
                json.dump(streams_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving CCTV database: {e}")
    
    def add_stream(self, stream_name, rtsp_url, location, lat=None, lng=None, start_monitoring=True):
        """Add a new RTSP stream"""
        logger.info(f"Adding stream: {stream_name}")
        
        with self.lock:
            # Check if exists and looks active
            # Check if exists
            if stream_name in self.active_streams:
                logger.info(f"Stream {stream_name} already exists. Updating configuration...")
                # Mark old stream as inactive to stop its thread
                self.active_streams[stream_name]['active'] = False
                # Brief pause to allow thread to notice? ideally not needed if we overwrite active_streams entry next
                # But to be safe we can just let it be overwritten, the old thread has a local ref to the old dict output
                # The old dict's 'active' key is now False, so the thread will exit loop.
                pass

            self.active_streams[stream_name] = {
                'url': rtsp_url,
                'location': location,
                'lat': lat,
                'lng': lng,
                'active': start_monitoring,
            'status': 'disabled' if not start_monitoring else 'connecting',
                'last_update': datetime.now(),
                'added_date': datetime.now().isoformat(),
                'error_count': 0
            }
        
        if start_monitoring:
            self.start_stream_monitoring(stream_name)
        
        self.save_streams_to_db()
        return True
    
    def add_webcam_stream(self, stream_name="Live Webcam", location="Your Location", active=False):
        """Add webcam as a stream for testing"""
        # Cleanup existing stream if present
        self.stop_stream(stream_name)
        
        # Test if webcam is available
        try:
            if active:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        return self.add_stream(stream_name, "0", location, start_monitoring=True)
            else:
                # Add as disabled immediately
                with self.lock:
                    self.blocked_streams.add(stream_name)
                    logger.info(f"Adding webcam {stream_name} as DISABLED (Blocked by default)")
                return self.add_stream(stream_name, "0", location, start_monitoring=False)
            
            logger.error("Webcam not accessible or active=False check skipped")
            return False
        except Exception as e:
            logger.error(f"Error testing webcam: {e}")
            return False

    def __init__(self, config=None):
        self.config = config
        self.active_streams = {}
        self.stream_threads = {}
        self.stream_readers = {} # New: Track readers for forced cleanup
        self.blocked_streams = set() # New: Explicit block list (Kill Switch)
        self.latest_frames = {}
        self.latest_detections = {}
        self.lock = threading.Lock()
        self.running = True
        
        # Load configuration
        if self.config:
            self.lost_faces_dir = os.path.join(self.config.UPLOAD_FOLDER, 'persons')
            os.makedirs(self.lost_faces_dir, exist_ok=True)
        else:
            # Fallback for direct instantiation without config (should not happen in app)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.lost_faces_dir = os.path.join(base_dir, '..', 'data', 'uploads', 'persons')
            
        self.load_streams_from_db()

        # Load lost persons for matching
        self.lost_face_encodings = []
        self.lost_face_names = []
        self.load_lost_persons_database() # Assuming this is the correct method name
        
        # Detection throttling
        self.last_detection_times = {} # Key: (stream_name, person_name), Value: timestamp

    def stop_stream(self, stream_name, remove_from_config=True):
        """Stop a specific stream monitoring"""
        logger.warning(f"🛑 STOPPING STREAM REQUEST: {stream_name}")
        
        # 0. BLOCK IT immediately if this is a toggle-off (not a delete)
        if not remove_from_config:
            with self.lock:
                self.blocked_streams.add(stream_name)
                logger.warning(f"🚫 BLOCKED stream {stream_name} (Kill Switch Active)")
        
        # 1. First, mark as inactive to signal thread loop to stop
        with self.lock:
            if stream_name in self.active_streams:
                self.active_streams[stream_name]['active'] = False
                logger.warning(f"Set active=False for {stream_name}")
            else:
                logger.warning(f"{stream_name} not found in active_streams during stop")
        
        # 2. THEN Force release the reader to interrupt any blocking I/O
        if stream_name in self.stream_readers:
            try:
                reader = self.stream_readers[stream_name]
                logger.warning(f"FORCE RELEASING reader for {stream_name}")
                reader.release()
                del self.stream_readers[stream_name]
            except Exception as e:
                logger.error(f"Error force releasing reader: {e}")
        
        # 3. Wait for thread to finish
        if stream_name in self.stream_threads:
            thread = self.stream_threads[stream_name]
            logger.warning(f"Waiting for thread {stream_name} to join...")
            if thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    logger.error(f"❌ Thread {stream_name} DID NOT DIE after 3s join")
                else:
                    logger.warning(f"✅ Thread {stream_name} joined successfully")
            del self.stream_threads[stream_name]
            
        with self.lock:
            if stream_name in self.active_streams:
                if remove_from_config:
                    del self.active_streams[stream_name]
                else:
                    self.active_streams[stream_name]['status'] = 'disabled'
                    
        self.save_streams_to_db()

    def set_stream_active(self, stream_name, active):
        """Enable or disable a stream"""
        if active:
            # UNBLOCK if explicitly enabled
            with self.lock:
                if stream_name in self.blocked_streams:
                    self.blocked_streams.remove(stream_name)
                    logger.warning(f"✅ UNBLOCKED stream {stream_name} (User explicitly enabled)")

            if stream_name not in self.active_streams:
                return False
                
            with self.lock:
                self.active_streams[stream_name]['active'] = True
                
            return self.start_stream_monitoring(stream_name)
        else:
            self.stop_stream(stream_name, remove_from_config=False)
            return True

    def test_rtsp_connection(self, rtsp_url):
        """Test if RTSP stream is accessible"""
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            cap.release()
            return ret
        except Exception:
            return False
    
    def start_stream_monitoring(self, stream_name):
        """Start monitoring a specific stream"""
        # CHECK BLOCK LIST
        with self.lock:
            if stream_name in self.blocked_streams:
                logger.warning(f"⛔ Cannot start stream {stream_name}: BLOCKED (Kill Switch Active)")
                return False

        if stream_name in self.stream_threads and self.stream_threads[stream_name].is_alive():
             return True
        
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
        """Monitor stream loop"""
        stream_info = None
        with self.lock:
             stream_info = self.active_streams.get(stream_name)
        
        if not stream_info: return

        # Initialize capture source
        reader = None
        reconnect_interval = 5  # seconds
        
        # Set initial status
        with self.lock:
            stream_info['status'] = 'connecting'
            
        frame_count = 0 
        
        try:
            while self.running and stream_name in self.active_streams and self.active_streams[stream_name]['active']:
                # CHECK BLOCK LIST
                if stream_name in self.blocked_streams:
                    logger.warning(f"⛔ Breaking loop for {stream_name}: BLOCKED")
                    break

                try:
                    # Initialize or Reconnect
                    if reader is None or not reader.isOpened():
                        with self.lock:
                            stream_info['status'] = 'connecting'
                        
                        # Connection Logic (Transport Options)
                        # Force TCP for RTSP (better stability), but NOT for RTMP or port 1935 (non-standard)
                        if stream_info['url'].lower().startswith('rtsp://') and ':1935' not in stream_info['url']:
                            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                        elif stream_info['url'].lower().startswith('rtmp://') or ':1935' in stream_info['url']:
                            if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                                del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]

                        # Start Reader Thread
                        if reader: 
                            # Remove from global tracking if replacing
                            if stream_name in self.stream_readers:
                                del self.stream_readers[stream_name]
                            reader.release()
                        
                        # DOUBLE CHECK ACTIVE STATUS before creating reader to prevent 'pop'
                        if not self.active_streams[stream_name].get('active', False):
                            logger.warning(f"Aborting reader creation for {stream_name} - stream inactive")
                            break

                        reader = StreamReader(stream_info['url'])
                        
                        # Track it globally
                        self.stream_readers[stream_name] = reader

                        
                        time.sleep(1) # Give it a moment to connect
                        
                        if not reader.isOpened():
                             # Auto-Healing Logic
                             logger.warning(f"Stream {stream_name} failed to open. Attempting auto-port scan...")
                             new_url = self._scan_and_fix_url(stream_info['url'])
                             if new_url and new_url != stream_info['url']:
                                   logger.info(f"Auto-Healing: Found open port! Switching {stream_info['url']} -> {new_url}")
                                   with self.lock:
                                       stream_info['url'] = new_url 
                                   reader = StreamReader(new_url)
                                   self.save_streams_to_db()

                        if not reader.isOpened():
                               logger.warning(f"Failed to open stream {stream_name}. Retrying in {reconnect_interval}s...")
                               with self.lock:
                                   stream_info['status'] = 'error'
                               time.sleep(reconnect_interval)
                               continue
                        else:
                            logger.info(f"Successfully connected to {stream_name}")
                            with self.lock:
                                stream_info['status'] = 'online'

                    # Read Latest Frame
                    frame = reader.read()
                    
                    if frame is None:
                        if not reader.running or reader.status == 'error':
                             logger.warning(f"Reader failed for {stream_name}. Reconnecting...")
                             reader.release()
                             reader = None
                             continue
                        else:
                            time.sleep(0.1)
                            continue

                    # Resize
                    if frame.shape[1] != 640 or frame.shape[0] != 480:
                        frame = cv2.resize(frame, (640, 480))
                    
                    # Detection
                    frame_count += 1
                    if frame_count % 5 == 0:
                        try:
                            if hasattr(self.config, 'face_matcher') and self.config.face_matcher:
                                matches = self.config.face_matcher.detect_and_match_faces_realtime(
                                    frame,
                                    self.lost_face_encodings,
                                    self.lost_face_names,
                                    threshold=0.5
                                )
                                if matches:
                                    logger.debug(f"Matches found: {len(matches)} in {stream_name}")
                                else:
                                    # logger.debug(f"No matches in {stream_name}") 
                                    pass

                                with self.lock:
                                    self.latest_detections[stream_name] = matches
                                    
                                for match in matches:
                                    if match['found']:
                                        raw_name = match['name']
                                        # Resolve display name for logging
                                        display_name = raw_name
                                        try:
                                            from utils.helpers import load_persons_from_db
                                            persons = load_persons_from_db(self.config.PERSONS_DB_FILE)
                                            if raw_name in persons:
                                                display_name = persons[raw_name].get('display_name', raw_name)
                                        except: pass

                                        logger.info(f"🚨 FOUND: {display_name} ({raw_name}) in {stream_name} ({match['similarity']:.2f})")
                                        try:
                                            from utils.helpers import save_detection_to_db
                                            detection_key = (stream_name, raw_name)
                                            current_time = time.time()
                                            last_time = self.last_detection_times.get(detection_key, 0)
                                            
                                            if current_time - last_time > 60:
                                                if hasattr(self.config, 'DETECTIONS_DB_FILE'):
                                                    record = {
                                                        'person_name': raw_name, # Process ID for DB consistency
                                                        'display_name': display_name, # Save display name for easier access (optional)
                                                        'similarity': float(match['similarity']),
                                                        'stream_name': stream_name,
                                                        'timestamp': datetime.now().isoformat(),
                                                        'location': stream_info.get('location', 'Unknown')
                                                    }
                                                    save_detection_to_db(record, self.config.DETECTIONS_DB_FILE)
                                                    self.last_detection_times[detection_key] = current_time
                                            else:
                                                pass
                                        except: pass
                        except Exception as e:
                            logger.error(f"Inference error: {e}")

                    if frame_count % 100 == 0:
                        logger.debug(f"Stream {stream_name} loop running. Active: {self.active_streams[stream_name]['active']}")

                    # Update Shared State
                    with self.lock: 
                        self.latest_frames[stream_name] = frame 
                        stream_info['last_update'] = datetime.now()
                        stream_info['error_count'] = 0 
                        stream_info['status'] = 'online'
                
                    time.sleep(0.01) # Reduced from 0.04 to minimize lag
                
                except Exception as inner_e:
                    logger.error(f"Error in stream monitor loop for {stream_name}: {inner_e}")
                    time.sleep(1)
        
        finally:
            logger.warning(f"🔄 EXITING _monitor_stream loop for {stream_name}")
            # Release resources strictly
            if reader is not None:
                reader.release()
                logger.info(f"Released stream resources for {stream_name}")
    
    def get_current_frame(self, stream_name, as_base64=False):
        """Get the latest frame, optionally with overlays enabled by caller or internal logic"""
        frame = None
        stream_status = 'unknown'
        
        with self.lock:
            frame = self.latest_frames.get(stream_name)
            if stream_name in self.active_streams:
                stream_status = self.active_streams[stream_name].get('status', 'unknown')
        
        if frame is None:
            # Return placeholder based on status
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            
            if stream_status == 'error':
                text = "Connection Failed"
                color = (0, 0, 255) # Red
                subtext = "Retrying..."
            elif stream_status == 'connecting':
                text = "Connecting..."
                color = (255, 165, 0) # Orange
                subtext = "Please wait"
            else:
                text = "Loading..."
                color = (255, 255, 255) # White
                subtext = "Initializing"
                
            cv2.putText(placeholder, text, (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(placeholder, subtext, (220, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
            
            frame = placeholder

        # --- Detection Logic ---
        # Only run detection if we have config and matchers available
        # And throttle it: we don't want to run detection on every single frame request if calls are frequent
        # But for now, we'll do it "realtime" per request or rely on the stream thread?
        # A better architecture is: Stream thread updates 'latest_frame'. 
        # A separate Detection thread (or the same stream thread) updates 'latest_overlays'.
        # For simplicity in this fix, we will run detection ON DEMAND here but with a throttle check?
        # Actually, running detection inside get_current_frame slows down serving. 
        # Let's run detection here but optimizing simply.
        
        # --- optimized Detection Logic ---
        # Draw cached detections from background thread
        if hasattr(self.config, 'face_matcher') and self.config.face_matcher:
            with self.lock:
                matches = self.latest_detections.get(stream_name, [])
            
            for match in matches:
                # Draw ALL detected faces
                x1, y1, x2, y2 = map(int, match['bbox'])
                raw_name = match.get('name', 'Unknown')
                confidence = match.get('similarity', 0.0)
                is_found = match.get('found', False)
                
                # Resolve display name for overlay
                display_name = raw_name
                
                # 1. Try DB Lookup
                try:
                    from utils.helpers import load_persons_from_db
                    persons = load_persons_from_db(self.config.PERSONS_DB_FILE)
                    
                    found_in_db = False
                    # Direct check failed? Try checking image_paths
                    for p_id, p_data in persons.items():
                        p_img_path = p_data.get('image_path', '')
                        if not p_img_path: continue
                        
                        # Extract basename without extension
                        p_filename = os.path.basename(p_img_path)
                        p_stem = os.path.splitext(p_filename)[0]
                        
                        # Check complete match
                        if p_stem == raw_name:
                            display_name = p_data.get('display_name', p_data.get('name', raw_name))
                            found_in_db = True
                            break
                            
                    if not found_in_db:
                         if raw_name in persons:
                            display_name = persons[raw_name].get('display_name', raw_name)
                            
                except: pass
                
                # 3. Fallback: Strip UUID if it looks like one (contains underscore)
                if display_name == raw_name and '_' in display_name:
                    parts = display_name.split('_', 1)
                    if len(parts) > 1:
                        display_name = parts[1] # Take the part after the first underscore

                if is_found:
                    color = (0, 255, 0) # Green for match
                    label = f"{display_name} ({confidence*100:.0f}%)"
                else:
                    display_name = "Unknown"
                    color = (0, 0, 255) # Red for unknown
                    label = "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                               
        if as_base64:
             ret, buffer = cv2.imencode('.jpg', frame)
             if ret:
                 return base64.b64encode(buffer).decode('utf-8')
             return None
        else:
             ret, buffer = cv2.imencode('.jpg', frame)
             if ret:
                 return buffer.tobytes()
             return None

    def get_stream_status(self):
        with self.lock:
             return {name: info.copy() for name, info in self.active_streams.items()}

    def stop_all_streams(self):
        self.running = False
        for name in list(self.stream_threads.keys()):
            self.stop_stream(name)
            
    def _scan_and_fix_url(self, url):
        """Scan common camera ports and return fixed URL if open port found"""
        try:
            import socket
            from urllib.parse import urlparse, urlunparse
            
            parsed = urlparse(url)
            if not parsed.hostname: return None
            
            # Common camera ports to check
            common_ports = [554, 1935, 8080, 8554, 80, 5000]
            current_port = parsed.port
            
            # Remove current port from list to avoid redundancy
            if current_port in common_ports:
                try:
                    common_ports.remove(current_port)
                except ValueError:
                    pass
                
            logger.info(f"Scanning ports {common_ports} for {parsed.hostname}...")
            
            for port in common_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5) # Fast scan
                result = sock.connect_ex((parsed.hostname, port))
                sock.close()
                
                if result == 0:
                    logger.info(f"Port {port} is OPEN!")
                    # Construct new URL
                    # Determine scheme based on port
                    scheme = 'rtsp'
                    if port == 1935: scheme = 'rtmp'
                    elif port in [80, 8080]: scheme = 'http' # Just a guess, but better than nothing
                    
                    # Rebuild URL
                    netloc = f"{parsed.username}:{parsed.password}@{parsed.hostname}:{port}" if parsed.username else f"{parsed.hostname}:{port}"
                    parts = list(parsed)
                    parts[0] = scheme
                    parts[1] = netloc
                    return urlunparse(parts)
            
            return None
        except Exception as e:
            logger.error(f"Auto-healing failed: {e}")
            return None
        
        