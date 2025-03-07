import cv2
import numpy as np
import torch
import threading
import time
import os
import json
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict
from twilio.rest import Client
from dotenv import load_dotenv
from playsound import playsound
from models.action_recognition import ActionRecognitionModel

# Load environment variables
load_dotenv()

class SurveillanceSystem:
    def __init__(self, config_path: str = "config/surveillance_config.json"):
        """Initialize the surveillance system
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = ActionRecognitionModel()
        self.frame_buffer = deque(maxlen=self.config["frame_buffer_size"])
        self.alert_cooldown = timedelta(seconds=self.config["alert_cooldown"])
        self.last_alert_time = datetime.now() - self.alert_cooldown
        self.recording = False
        self.setup_twilio()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "frame_buffer_size": 16,  # Number of frames to analyze at once
            "alert_cooldown": 30,     # Seconds between alerts
            "confidence_threshold": 0.7,
            "suspicious_actions": [    # Actions that trigger alerts
                "fighting",
                "falling",
                "vandalism",
                "throwing",
                "stealing",
                "breaking",
                "attacking"
            ],
            "recording": {
                "enabled": True,
                "pre_event": 5,    # Seconds to keep before event
                "post_event": 10,  # Seconds to keep after event
                "output_dir": "recordings"
            },
            "display": {
                "show_fps": True,
                "show_predictions": True,
                "show_alerts": True
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def setup_twilio(self):
        """Setup Twilio client for SMS alerts"""
        self.twilio_client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        self.target_phone = os.getenv("TARGET_PHONE_NUMBER")
    
    def send_alert(self, activity: str, confidence: float):
        """Send alert through multiple channels"""
        current_time = datetime.now()
        if (current_time - self.last_alert_time) < self.alert_cooldown:
            return
            
        # Play alarm sound
        threading.Thread(target=self._play_alarm).start()
        
        # Send SMS
        self._send_sms(f"⚠️ Alert: {activity} detected with {confidence:.2f}% confidence at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Start recording if enabled
        if self.config["recording"]["enabled"] and not self.recording:
            self._start_recording()
        
        self.last_alert_time = current_time
    
    def _play_alarm(self):
        """Play alarm sound"""
        try:
            playsound("assets/alarm.wav")
        except Exception as e:
            print(f"Error playing alarm: {e}")
    
    def _send_sms(self, message: str):
        """Send SMS alert"""
        try:
            self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_phone,
                to=self.target_phone
            )
        except Exception as e:
            print(f"Error sending SMS: {e}")
    
    def _start_recording(self):
        """Start recording video"""
        self.recording = True
        threading.Thread(target=self._record_video).start()
    
    def _record_video(self):
        """Record video of the event"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.config["recording"]["output_dir"],
            f"event_{timestamp}.mp4"
        )
        
        # Ensure output directory exists
        os.makedirs(self.config["recording"]["output_dir"], exist_ok=True)
        
        # Initialize video writer
        frame = self.frame_buffer[0] if self.frame_buffer else None
        if frame is not None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
            
            # Write buffered frames
            for frame in self.frame_buffer:
                out.write(frame)
            
            # Continue recording for post_event duration
            end_time = time.time() + self.config["recording"]["post_event"]
            while time.time() < end_time:
                if len(self.frame_buffer) > 0:
                    out.write(self.frame_buffer[-1])
                time.sleep(1/30)
            
            out.release()
        
        self.recording = False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame
        
        Args:
            frame: Input frame to process
            
        Returns:
            Processed frame with visualizations
        """
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # Process frames if buffer is full
        if len(self.frame_buffer) == self.config["frame_buffer_size"]:
            predictions = self.model.process_video_segment(list(self.frame_buffer))
            
            # Clear half of the buffer to allow overlap between segments
            for _ in range(self.config["frame_buffer_size"] // 2):
                self.frame_buffer.popleft()
            
            # Check for suspicious activities
            for action, confidence in predictions.items():
                if (action in self.config["suspicious_actions"] and 
                    confidence > self.config["confidence_threshold"]):
                    self.send_alert(action, confidence * 100)
            
            # Add visualizations
            if self.config["display"]["show_predictions"]:
                # Show top predictions
                y_pos = 30
                # Get all predictions and sort by confidence
                all_predictions = [(k, v) for k, v in predictions.items()]
                all_predictions.sort(key=lambda x: x[1], reverse=True)
                
                # Show predictions
                for action, conf in all_predictions[:2]:  # Show top 2 predictions
                    text = f"{action}: {conf*100:.1f}%"
                    cv2.putText(frame, text, (10, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_pos += 30
        
        return frame
    
    def run(self, camera_id: int = 0):
        """Run the surveillance system
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        start_time = time.time()
        frame_count = 0
        
        print("\nSurveillance System Started")
        print("Press 'q' to quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Calculate and display FPS
            if self.config["display"]["show_fps"]:
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Surveillance System", processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create surveillance system
    system = SurveillanceSystem()
    
    # Run the system
    system.run() 