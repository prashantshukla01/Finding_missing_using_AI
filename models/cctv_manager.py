import cv2
import threading
import time
import logging
import json
import base64
import os
from datetime import datetime
from queue import Queue
import numpy as np

logger = logging.getLogger(__name__)

class CCTVManager:
    def __init__(self, config):
        self.config = config
        self.active_streams = {}
        self.frame_queues = {}
        self.stream_threads = {}
        self.running = False
        
        # Load existing streams from database
        self.load_streams_from_db()
        logger.info("CCTV Manager initialized successfully")
    
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
    
    def add_webcam_stream(self, stream_name="Webcam", location="Local"):
        """Add webcam as a stream for testing"""
        logger.info(f"Attempting to add webcam stream: {stream_name}")
        
        # Use 0 for default webcam
        webcam_url = "0"
        
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
                logger.error("Webcam cannot be opened")
        except Exception as e:
            logger.error(f"Error testing webcam: {e}")
        
        return False
    
    def test_rtsp_connection(self, rtsp_url):
        """Test if RTSP stream is accessible"""
        try:
            logger.info(f"Testing RTSP connection: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            
            if not cap.isOpened():
                return False
            
            # Try to read a frame with timeout
            start_time = time.time()
            while (time.time() - start_time) < 10:
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return True
                time.sleep(0.1)
            
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
        """Monitor stream and capture frames"""
        stream_info = self.active_streams[stream_name]
        
        while self.running and stream_info['active']:
            try:
                # Create a test frame for demonstration
                test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
                cv2.putText(test_image, f"Stream: {stream_name}", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(test_image, "Face Detection System", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(test_image, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (50, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Update stream info
                stream_info['last_frame'] = test_image
                stream_info['last_update'] = datetime.now()
                
                # Put frame in queue
                if not self.frame_queues[stream_name].empty():
                    try:
                        self.frame_queues[stream_name].get_nowait()
                    except:
                        pass
                
                self.frame_queues[stream_name].put(test_image)
                
                time.sleep(2)  # Update every 2 seconds for demo
                
            except Exception as e:
                logger.error(f"Error in stream monitoring for {stream_name}: {e}")
                time.sleep(5)
    
    def get_current_frame(self, stream_name, as_base64=False):
        """Get current frame from stream"""
        if stream_name not in self.frame_queues:
            return None
        
        try:
            frame = self.frame_queues[stream_name].get_nowait()
            
            if as_base64 and frame is not None:
                # Convert frame to base64 for web display
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                return frame_base64
            
            return frame
            
        except:
            # Return a default frame if queue is empty
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, f"Stream: {stream_name}", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(test_image, "No frame available", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            if as_base64:
                _, buffer = cv2.imencode('.jpg', test_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                return frame_base64
            
            return test_image
    
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