
import cv2
import threading
import time
import logging
from functools import wraps # Corrected typo from 'impofrom'
import json
import base64
import numpy as np
import os
from datetime import datetime
from queue import Queue, Empty
from app.models.db_models import db, Detection, Person # Assuming PersonEmpty was a typo for Person
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
        # Handle webcam index
        if str(url).isdigit():
            self.cap = cv2.VideoCapture(int(url))
        elif str(url).lower() in ['demo', 'webcam', 'camera']:
            logger.info(f"Opening default camera for stream URL: {url}")
            self.cap = cv2.VideoCapture(0)
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
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.q.append(frame)
                    with self.lock: self.status = 'online'
                else:
                    # If reading fails for a normal stream, it's an error
                    if str(self.url).lower() in ['demo', 'webcam']:
                        # Fallback for demo: Generate synthetic noise
                        synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        cv2.putText(synthetic_frame, "DEMO MODE (No Camera)", (150, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        # Add timestamp
                        time_str = datetime.now().strftime("%H:%M:%S")
                        cv2.putText(synthetic_frame, time_str, (500, 450), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        self.q.append(synthetic_frame)
                        with self.lock: self.status = 'online'
                        time.sleep(0.1) # Simulate FPS
                    else:
                        with self.lock: self.status = 'error'
                        time.sleep(0.1)
            else:
                # If capture not opened
                if str(self.url).lower() in ['demo', 'webcam']:
                     # Synthetic fallback immediately
                     with self.lock: self.status = 'online'
                     synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                     cv2.putText(synthetic_frame, "NO WEBCAM FOUND", (180, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                     self.q.append(synthetic_frame)
                     time.sleep(1)
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
        if self.cap and self.cap.isOpened():
            return True
        # Critical Fallback: Enable synthetic frames if hardware cam fails
        # Usage: digits (0, 1) or 'demo'/'webcam' keywords
        if str(self.url).lower() in ['demo', 'webcam', 'camera'] or str(self.url).isdigit():
            return True
        return False
        
    def reconnect(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.url)

class CCTVManager:
    def __init__(self, config=None, app=None):
        self.config = config
        self.app = app
        self.active_streams = {}
        self.latest_frames = {}
        self.latest_detections = {}
        self.lock = threading.Lock()
        
        self.person_name_map = {}
        
        self.stream_readers = {}
        self.stream_threads = {}
        self.blocked_streams = set()
        self.running = True
        self.last_detection_times = {} 
        
        # Lost persons database
        if self.config:
            self.lost_faces_dir = os.path.join(self.config.UPLOAD_FOLDER, 'persons')
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.lost_faces_dir = os.path.join(base_dir, '..', 'data', 'uploads', 'persons')
        
        os.makedirs(self.lost_faces_dir, exist_ok=True)
        
        self.lost_face_encodings = []
        self.lost_face_names = []
        
        # Initialize
        self.load_streams_from_db()
        # Initialize
        self.load_streams_from_db()
        self.reload_lost_persons_database()
        
        logger.info("CCTV Manager initialized successfully")
        
        logger.info("CCTV Manager initialized successfully")

    def load_streams_from_db(self):
        """Load streams from database on initialization"""
        try:
            # Ensure we're in app context
            if self.app:
                with self.app.app_context():
                    from app.models.db_models import Stream
                    streams = Stream.query.all()
                    
                    for stream in streams:
                        # POLICY: Webcams must start OFF by default for privacy/battery
                        # Check digits OR keywords
                        src_lower = str(stream.source_url).lower()
                        is_webcam = src_lower.isdigit() or src_lower in ['demo', 'webcam', 'camera']
                        
                        # FORCE OFF for webcams on startup, regardless of DB state
                        if is_webcam:
                            initial_active = False 
                            if stream.active:
                                logger.info(f"🔒 Security: Forcing stream '{stream.name}' OFF on startup.")
                                stream.active = False
                                db.session.commit()
                        else:
                            initial_active = stream.active
                        
                        # Initialize active_streams dictionary entry first
                        self.active_streams[stream.name] = {
                            'name': stream.name,
                            'url': stream.source_url,
                            'status': 'offline', # Start offline, let monitor set it
                            'active': initial_active,
                            'location': stream.location # Add location for UI
                        }
                        
                        if initial_active:
                            logger.info(f"Starting stream from DB: {stream.name}")
                            # Correct way: use start_stream_monitoring
                            self.start_stream_monitoring(stream.name)
                        else:
                            status_msg = "Force-Disabled Webcam" if is_webcam else "Inactive"
                            logger.info(f"Loaded {status_msg} stream from DB: {stream.name}")
                            
        except Exception as e:
            logger.error(f"Error loading streams from DB: {str(e)}")

    def reload_lost_persons_database(self):
        """Load encodings for lost persons from the uploads directory"""
        try:
            if self.app:
                with self.app.app_context():
                    from app.models.db_models import Person
                    persons = Person.query.all()
                    self.lost_face_encodings = []
                    self.lost_face_names = []
                    
                    count = 0
                    for p in persons:
                        if p.embedding:
                            # CRITICAL FIX: The embedding loaded from DB is a Dictionary containing 'insightface' key
                            # We must extract the vector part for the matcher
                            if isinstance(p.embedding, dict) and 'insightface' in p.embedding:
                                vec = np.array(p.embedding['insightface'])
                                self.lost_face_encodings.append(vec)
                                self.lost_face_names.append(p.name)
                                count += 1
                                # logger.info(f"Loaded embedding for {p.name}: shape {vec.shape}")
                            elif isinstance(p.embedding, (list, np.ndarray)):
                                # Fallback if somehow stored as direct list
                                self.lost_face_encodings.append(np.array(p.embedding))
                                self.lost_face_names.append(p.name)
                                count += 1
                            else:
                                logger.warning(f"Skipping invalid embedding format for {p.name}")

            logger.info(f"Loaded {count} lost persons from database")
                        
        except Exception as e:
            logger.error(f"Error loading lost persons database: {str(e)}")

    # Implemented using DB persistence
    def add_stream(self, name, url, location, lat=None, lng=None):
        """Add a new stream"""
        if self.app:
            try:
                with self.app.app_context():
                    from app.models.db_models import Stream
                    # Create in DB
                    new_stream = Stream(
                        name=name,
                        source_url=url,
                        location=location,
                        lat=lat,
                        lng=lng,
                        active=True # Default to TRUE for better UX
                    )
                    db.session.add(new_stream)
                    db.session.commit()
                    
                    # Update active_streams
                    self.active_streams[name] = {
                        'name': name,
                        'url': url,
                        'location': location,
                        'status': 'offline',
                        'active': True
                    }
                    
                    # Start monitoring immediately
                    self.start_stream_monitoring(name)
                    return True
            except Exception as e:
                logger.error(f"Error adding stream: {e}")
                return False
        return False

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
            # FIX: Clear the latest frame to prevent stale image
            if stream_name in self.latest_frames:
                del self.latest_frames[stream_name]
            
            # FIX: Clear latest detections to prevent stale overlays
            if stream_name in self.latest_detections:
                del self.latest_detections[stream_name]

            if stream_name in self.active_streams:
                if remove_from_config:
                    del self.active_streams[stream_name]
                else:
                    self.active_streams[stream_name]['status'] = 'disabled'
        
        # FIX: Explicitly handle Database Deletion
        if remove_from_config and self.app:
            try:
                with self.app.app_context():
                    from app.models.db_models import Stream
                    stream_to_delete = Stream.query.filter_by(name=stream_name).first()
                    if stream_to_delete:
                        db.session.delete(stream_to_delete)
                        db.session.commit()
                        logger.info(f"🗑️ Deleted stream '{stream_name}' from database")
                    else:
                        logger.warning(f"Stream '{stream_name}' not found in DB for deletion")
            except Exception as e:
                logger.error(f"Error deleting stream from DB: {e}")
        
        if self.config:
            self.load_streams_from_db()
                    
        # self.save_streams_to_db() # Removed non-existent method call

    def set_stream_active(self, stream_name, active):
        """Enable or disable a stream"""
        # 1. Update DB Persistently
        if self.app:
            try:
                with self.app.app_context():
                    from app.models.db_models import Stream
                    stream = Stream.query.filter_by(name=stream_name).first()
                    if stream:
                        stream.active = active
                        db.session.commit()
                        logger.info(f"Updated DB status for {stream_name} to {active}")
            except Exception as e:
                logger.error(f"Error updating stream status in DB: {e}")

        if active:
            # UNBLOCK if explicitly enabled
            with self.lock:
                if stream_name in self.blocked_streams:
                    self.blocked_streams.remove(stream_name)
                    logger.warning(f"✅ UNBLOCKED stream {stream_name} (User explicitly enabled)")

            if stream_name not in self.active_streams:
                # If not in memory but valid, try loading it? 
                # Or just return False
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
                                        # Resolve display name for logging using in-memory map
                                        display_name = self.person_name_map.get(raw_name, raw_name)

                                        logger.info(f"🚨 FOUND: {display_name} ({raw_name}) in {stream_name} ({match['similarity']:.2f})")
                                        
                                        try:
                                            detection_key = (stream_name, raw_name)
                                            current_time = time.time()
                                            last_time = self.last_detection_times.get(detection_key, 0)
                                            
                                            if current_time - last_time > 5:
                                                if self.app:
                                                    try:
                                                        with self.app.app_context():
                                                            from app.models.db_models import Detection, db
                                                            
                                                            new_detection = Detection(
                                                                person_name=raw_name,
                                                                stream_name=stream_name,
                                                                confidence=float(match['similarity']),
                                                                timestamp=datetime.now()
                                                            )
                                                            db.session.add(new_detection)
                                                            db.session.commit()
                                                    except Exception as e:
                                                        logger.error(f"Failed to save detection to DB: {e}")
                                                        
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
                # Resolve display name for overlay
                display_name = self.person_name_map.get(raw_name, raw_name)
                
                # 3. Fallback: Strip UUID if it looks like one (contains underscore)
                
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

    def get_frame_generator(self, stream_name):
        """Yield frames for MJPEG streaming"""
        while self.running:
            jpeg_bytes = self.get_current_frame(stream_name, as_base64=False)
            
            if jpeg_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            else:
                 time.sleep(0.1)
                 
            time.sleep(0.04) # Limit FPS

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
        
        