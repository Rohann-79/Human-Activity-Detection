import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pytorchvideo.models.resnet
import numpy as np
from typing import Dict, List
import json
import os
from collections import deque
import time

class ActionRecognitionModel:
    def __init__(self, model_name: str = "i3d_r50", num_classes: int = 400):
        """Initialize the action recognition model"""
        # Force CPU usage for M1 Mac compatibility
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize temporal smoothing
        self.prediction_history = deque(maxlen=5)  # Keep last 5 predictions
        self.confidence_threshold = 0.01  # Very low threshold for demo
        
        # Store positions for movement and fall detection
        self.shoulder_positions = deque(maxlen=10)  # Store last 10 x positions
        self.shoulder_heights = deque(maxlen=10)    # Store last 10 y positions
        self.last_fall_detection = 0  # Time of last fall detection
        
        # Focus on activities including falling
        self.relevant_activities = {
            'sitting',
            'standing',
            'walking',
            'falling'
        }
        
        # Load model and transforms
        self.model = self._load_model(model_name, num_classes)
        self.transform = self._get_transform()
        self.class_names = self._load_class_names()
        print(f"Loaded {len(self.class_names)} activity classes")
        print("Monitoring for activities:", list(self.relevant_activities))
        
    def _load_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Load the PyTorchVideo model"""
        try:
            # Use I3D model which is better for close-up views
            model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
            print("Successfully loaded I3D model")
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading I3D model: {e}")
            print("Falling back to Slow R50 model...")
            try:
                model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
                model = model.to(self.device)
                model.eval()
                return model
            except Exception as e:
                print(f"Error loading fallback model: {e}")
                return None
    
    def _get_transform(self) -> transforms.Compose:
        """Get video preprocessing transforms"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_class_names(self) -> List[str]:
        """Load action class names"""
        try:
            kinetics_labels = torch.hub.load('facebookresearch/pytorchvideo', 'kinetics_400_labels')
            # Save labels to file for reference
            with open("models/kinetics400_classes.json", "w") as f:
                json.dump(kinetics_labels, f, indent=4)
            return kinetics_labels
        except:
            if os.path.exists("models/kinetics400_classes.json"):
                with open("models/kinetics400_classes.json", "r") as f:
                    return json.load(f)
            return []
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame"""
        try:
            frame_tensor = self.transform(frame)
            return frame_tensor.to(self.device)
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None
    
    def _analyze_pose(self, frame: np.ndarray) -> str:
        """Analyze pose to determine if person is sitting, standing, walking, or falling"""
        results = self.pose.process(frame)
        if not results.pose_landmarks:
            return None
            
        landmarks = results.pose_landmarks.landmark
        
        # Get shoulder landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate shoulder position
        shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        
        # Store positions for movement and fall detection
        self.shoulder_positions.append(shoulder_x)
        self.shoulder_heights.append(shoulder_height)
        
        print(f"Debug - Shoulder height: {shoulder_height:.2f}, X position: {shoulder_x:.2f}")
        
        # Check for fall detection if we have enough frames
        if len(self.shoulder_heights) >= 5:
            # Calculate vertical velocity (change in height over frames)
            recent_heights = list(self.shoulder_heights)[-5:]
            vertical_velocity = (recent_heights[-1] - recent_heights[0]) / 5
            
            # Calculate acceleration (change in velocity)
            if len(recent_heights) >= 5:
                prev_velocity = (recent_heights[-2] - recent_heights[-5]) / 3
                acceleration = vertical_velocity - prev_velocity
                
                print(f"Debug - Vertical velocity: {vertical_velocity:.3f}, Acceleration: {acceleration:.3f}")
                
                # Detect fall based on:
                # 1. Rapid downward movement (positive velocity as y increases downward)
                # 2. Sudden acceleration
                # 3. Final position is low in frame
                current_time = time.time()
                if (vertical_velocity > 0.1 and  # Moving down rapidly
                    acceleration > 0.05 and      # Accelerating downward
                    shoulder_height > 0.7 and    # Ended up low in frame
                    current_time - self.last_fall_detection > 5):  # At least 5 seconds since last fall
                    
                    self.last_fall_detection = current_time
                    print("FALL DETECTED!")
                    return 'falling'
        
        # If no fall detected, check other activities
        if shoulder_height > 0.5:
            return 'sitting'
            
        # Check for walking
        if len(self.shoulder_positions) >= 5:
            x_positions = list(self.shoulder_positions)
            movement = max(x_positions) - min(x_positions)
            
            print(f"Debug - Movement amount: {movement:.3f}")
            
            if movement > 0.05:
                return 'walking'
            else:
                return 'standing'
                
        return 'standing'
    
    def process_video_segment(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """Process a segment of video frames and return action predictions"""
        if not frames or len(frames) < 8:
            return {}
            
        try:
            # Process frames for I3D
            processed_frames = []
            pose_predictions = []
            pose_confidence = 0.9  # Base confidence for pose detection
            
            for frame in frames[:16]:
                # Get pose prediction
                pose_pred = self._analyze_pose(frame)
                if pose_pred:
                    pose_predictions.append(pose_pred)
                    if pose_pred == 'falling':
                        # Immediately return falling with high confidence
                        return {'falling': 0.95}
                    
                # Process frame for I3D
                processed = self.preprocess_frame(frame)
                if processed is not None:
                    processed_frames.append(processed)
            
            if len(processed_frames) < 8:
                return {}
                
            # Get I3D predictions but ignore them for now
            clip = torch.stack(processed_frames).permute(1, 0, 2, 3)
            clip = clip.unsqueeze(0)
            
            with torch.no_grad():
                try:
                    i3d_predictions = self.model(clip)
                    
                    # Just use pose predictions
                    results = {}
                    
                    if pose_predictions:
                        # Count occurrences of each prediction
                        prediction_counts = {}
                        for pred in pose_predictions:
                            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
                        
                        # Calculate confidence based on consistency
                        total_preds = len(pose_predictions)
                        for pred, count in prediction_counts.items():
                            confidence = (count / total_preds) * pose_confidence
                            if pred not in results or confidence > results[pred]:
                                results[pred] = confidence
                    
                    # Sort by confidence and take top 1
                    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True)[:1])
                    
                    if sorted_results:
                        print("\nDetected activity:")
                        for activity, conf in sorted_results.items():
                            print(f"{activity}: {conf*100:.1f}%")
                    
                    return sorted_results
                    
                except Exception as e:
                    print(f"Error in model prediction: {e}")
                    return {}
                    
        except Exception as e:
            print(f"Error processing video segment: {e}")
            return {}
    
    def is_suspicious_activity(self, predictions: Dict[str, float], 
                             suspicious_actions: List[str], 
                             threshold: float = 0.5) -> bool:
        """Determine if the current activity is suspicious"""
        try:
            for action, confidence in predictions.items():
                if action in suspicious_actions and confidence > threshold:
                    return True
            return False
        except Exception as e:
            print(f"Error checking suspicious activity: {e}")
            return False 