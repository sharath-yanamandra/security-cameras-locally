import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from core_system import DataCenterMonitoringSystem

class ReceptionAreaMonitor:
    """
    Specialized monitor for reception area use cases:
    1. Crowd monitoring (not more than 10 people)
    2. Activity recognition for gate jumping
    """
    
    def __init__(self, system: DataCenterMonitoringSystem):
        """
        Initialize the reception area monitor.
        
        Args:
            system: The core monitoring system
        """
        self.system = system
        
        # Initialize activity recognition model (gate jumping)
        self.activity_detector = self._init_activity_detector()
        
        # History for tracking objects over time (for activity recognition)
        self.pose_history = {}
        
        # Variables for person counting
        self.current_count = 0
        self.person_positions = {}
        
    def _init_activity_detector(self):
        """Initialize activity detector for gate jumping detection."""
        # In a real implementation, we might load a specialized activity recognition model
        # For this implementation, we'll use pose estimation from the core system
        return None
    
    def count_people_in_reception(self, frame: np.ndarray, objects: List[Dict], 
                                reception_zone: Dict) -> Dict:
        """
        Count people in reception area and detect overcrowding.
        
        Args:
            frame: Input frame
            objects: Detected objects
            reception_zone: Zone definition for reception area
            
        Returns:
            Counting event with details
        """
        # Filter people in reception zone
        people_in_reception = [obj for obj in objects 
                             if obj['type'] == 'person' and 
                             self.system._is_in_zone(obj['bounding_box'], reception_zone)]
        
        # Count people
        person_count = len(people_in_reception)
        
        # Update current count
        self.current_count = person_count
        
        # Create counting event
        event = {
            'event_type': 'reception_count',
            'person_count': person_count,
            'zone_id': reception_zone['id'],
            'overcrowded': person_count > 10,  # Threshold for overcrowding
            'confidence': sum(p['confidence_score'] for p in people_in_reception) / max(1, person_count)
        }
        
        return event
    
    def detect_gate_jumping(self, frame: np.ndarray, objects: List[Dict], gate_zone: Dict) -> List[Dict]:
        """
        Detect people trying to jump/climb the entry gate.
        
        Args:
            frame: Input frame
            objects: Detected objects
            gate_zone: Zone definition for entry gate
            
        Returns:
            List of gate jumping events
        """
        jumping_events = []
        
        # Filter people near gate zone
        # Include a buffer around the gate to catch jumping attempts
        gate_buffer = 50  # pixels
        
        # Convert gate zone to extended bounding box
        gate_points = np.array(gate_zone['polygon'])
        min_x, min_y = np.min(gate_points, axis=0)
        max_x, max_y = np.max(gate_points, axis=0)
        
        # Extend the bounding box
        min_x -= gate_buffer
        min_y -= gate_buffer
        max_x += gate_buffer
        max_y += gate_buffer
        
        # Filter people near gate
        people_near_gate = []
        for obj in objects:
            if obj['type'] != 'person':
                continue
                
            # Get person bounding box center
            bbox = obj['bounding_box']
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            
            # Check if person is near gate
            if (min_x <= center_x <= max_x and min_y <= center_y <= max_y):
                people_near_gate.append(obj)
        
        # Process each person for activity recognition
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        for person in people_near_gate:
            person_id = person.get('tracking_id', '')
            
            if not person_id:
                continue
                
            # Get keypoints if available
            keypoints = person.get('metadata', {}).get('keypoints', [])
            
            if not keypoints or len(keypoints) < 17:  # Need full pose
                continue
                
            # Calculate key metrics for jumping detection
            
            # Get head, shoulder, hip, knee, ankle keypoints
            head = keypoints[0]  # Nose
            shoulders = [(keypoints[5][0] + keypoints[6][0])/2, 
                        (keypoints[5][1] + keypoints[6][1])/2]  # Mid-shoulders
            hips = [(keypoints[11][0] + keypoints[12][0])/2, 
                   (keypoints[11][1] + keypoints[12][1])/2]  # Mid-hips
            knees = [keypoints[13], keypoints[14]]  # Left and right knees
            ankles = [keypoints[15], keypoints[16]]  # Left and right ankles
            
            # Create pose feature vector
            pose_features = {
                'head_y': head[1],
                'shoulders_y': shoulders[1],
                'hips_y': hips[1],
                'knees_y': [k[1] for k in knees],
                'ankles_y': [a[1] for a in ankles],
                'timestamp': current_time
            }
            
            # Update pose history
            if person_id in self.pose_history:
                history = self.pose_history[person_id]
                history['poses'].append(pose_features)
                
                # Keep history to a reasonable size
                if len(history['poses']) > 30:  # About 5 seconds at 6 fps
                    history['poses'].pop(0)
                    
                # Analyze pose sequence for jumping
                if len(history['poses']) >= 10:  # Need enough history
                    is_jumping = self._detect_jumping_from_poses(history['poses'])
                    
                    if is_jumping and not history.get('jump_detected', False):
                        # New jumping event detected
                        history['jump_detected'] = True
                        history['jump_confidence'] = 0.8  # Initial confidence
                        
                        # Create jumping event
                        event = {
                            'event_type': 'gate_jumping',
                            'confidence': history['jump_confidence'],
                            'bounding_box': person['bounding_box'],
                            'zone_id': gate_zone['id'],
                            'object_id': person_id
                        }
                        jumping_events.append(event)
                    elif is_jumping and history.get('jump_detected', False):
                        # Update confidence for ongoing jump
                        history['jump_confidence'] = min(0.95, history['jump_confidence'] + 0.05)
                    elif not is_jumping:
                        # Reset jump detection
                        history['jump_detected'] = False
            else:
                # New person, initialize history
                self.pose_history[person_id] = {
                    'poses': [pose_features],
                    'jump_detected': False
                }
        
        # Clean up history for people that are no longer in the frame
        current_person_ids = [p.get('tracking_id', '') for p in people_near_gate]
        for person_id in list(self.pose_history.keys()):
            if person_id not in current_person_ids:
                # Person left the frame
                del self.pose_history[person_id]
                
        return jumping_events
    
    def _detect_jumping_from_poses(self, poses: List[Dict]) -> bool:
        """
        Analyze pose sequence to detect jumping.
        
        Args:
            poses: List of pose features over time
            
        Returns:
            True if jumping is detected, False otherwise
        """
        if len(poses) < 5:
            return False
            
        # Calculate vertical velocity of body parts
        head_positions = [p['head_y'] for p in poses]
        time_diffs = [poses[i+1]['timestamp'] - poses[i]['timestamp'] for i in range(len(poses)-1)]
        
        # Calculate vertical velocities
        head_velocities = [(head_positions[i] - head_positions[i+1]) / max(0.001, time_diffs[i]) 
                         for i in range(len(time_diffs))]
        
        # Check for jumping pattern: rapid upward movement followed by downward movement
        # This is a simplified heuristic - a real system would use a more sophisticated approach
        if len(head_velocities) >= 4:
            # Look for sequences where velocity goes from positive (upward) to negative (downward)
            # with significant magnitude
            velocity_threshold = 100  # Pixels per second
            
            for i in range(len(head_velocities) - 3):
                if (head_velocities[i] > velocity_threshold and
                    head_velocities[i+1] > velocity_threshold/2 and
                    head_velocities[i+2] < -velocity_threshold/2 and
                    head_velocities[i+3] < -velocity_threshold):
                    return True
        
        # Check knee extension (jumping typically involves extended knees)
        if len(poses) >= 3:
            # Calculate knee-ankle distance for each pose
            knee_ankle_dists = []
            for pose in poses:
                left_dist = abs(pose['knees_y'][0] - pose['ankles_y'][0])
                right_dist = abs(pose['knees_y'][1] - pose['ankles_y'][1])
                knee_ankle_dists.append(max(left_dist, right_dist))
            
            # Check for sudden increase in knee-ankle distance
            baseline = sum(knee_ankle_dists[:3]) / 3
            for dist in knee_ankle_dists[3:]:
                if dist > baseline * 1.5:  # 50% increase in distance
                    return True
        
        return False
    
    def process_frame(self, frame: np.ndarray, objects: List[Dict], zones: Dict) -> List[Dict]:
        """
        Process a frame for reception area use cases.
        
        Args:
            frame: Input frame
            objects: Detected objects
            zones: Zone definitions
            
        Returns:
            List of detected events
        """
        events = []
        
        # Get zone definitions
        reception_zone = next((z for z in zones if z['id'] == 'reception_waiting'), None)
        gate_zone = next((z for z in zones if z['id'] == 'entry_gate'), None)
        
        # Run detectors if zones are defined
        if reception_zone:
            counting_event = self.count_people_in_reception(frame, objects, reception_zone)
            events.append(counting_event)
            
        if gate_zone:
            jumping_events = self.detect_gate_jumping(frame, objects, gate_zone)
            events.extend(jumping_events)
            
        return events
        
    def annotate_frame(self, frame: np.ndarray, events: List[Dict]) -> np.ndarray:
        """
        Annotate the frame with detected events.
        
        Args:
            frame: Input frame
            events: Detected events
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for event in events:
            event_type = event.get('event_type', '')
            
            if event_type == 'reception_count':
                # Draw people count
                count = event['person_count']
                text = f"People Count: {count}"
                color = (0, 255, 0)  # Green by default
                
                if event.get('overcrowded', False):
                    text = f"ALERT: Overcrowded ({count} people)"
                    color = (0, 0, 255)  # Red for alert
                    
                cv2.putText(annotated, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            elif event_type == 'gate_jumping':
                # Draw red box around person jumping the gate
                bbox = event['bounding_box']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated, "ALERT: Gate Jumping", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        return annotated