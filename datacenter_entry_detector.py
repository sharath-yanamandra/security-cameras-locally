import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from core_system import DataCenterMonitoringSystem

class DataCenterEntryMonitor:
    """
    Specialized monitor for data center entry use cases:
    1. Unauthorized access detection
    2. Emergency button press detection
    """
    
    def __init__(self, system: DataCenterMonitoringSystem):
        """
        Initialize the data center entry monitor.
        
        Args:
            system: The core monitoring system
        """
        self.system = system
        
        # Keep track of previously detected people for unauthorized access
        self.authorized_ids = set()  # In a real system, this would be loaded from a database
        self.person_history = {}
        
        # For emergency button detection
        self.emergency_button_status = 'normal'  # 'normal' or 'pressed'
        self.button_press_time = None
        self.button_press_confidence = 0.0
    
    def detect_unauthorized_access(self, frame: np.ndarray, objects: List[Dict], 
                                 entry_door_zone: Dict) -> List[Dict]:
        """
        Detect unauthorized access to the data center.
        
        Args:
            frame: Input frame
            objects: Detected objects
            entry_door_zone: Zone definition for entry door
            
        Returns:
            List of unauthorized access events
        """
        unauthorized_events = []
        
        # Filter people in entry door zone
        people_in_entry = [obj for obj in objects 
                         if obj['type'] == 'person' and 
                         self.system._is_in_zone(obj['bounding_box'], entry_door_zone)]
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        for person in people_in_entry:
            person_id = person.get('tracking_id', '')
            
            if not person_id:
                continue
                
            # Check if person is already in history
            if person_id in self.person_history:
                # Update history
                history = self.person_history[person_id]
                history['last_seen'] = current_time
                history['positions'].append(person['bounding_box'])
                
                # Keep history at a reasonable size
                if len(history['positions']) > 30:
                    history['positions'].pop(0)
                    
                # Check if person is unauthorized
                if not history.get('authorized', False) and not history.get('alert_sent', False):
                    # Create unauthorized access event
                    # This would typically involve face recognition or badge detection
                    # For this example, we'll simulate by checking if ID is in authorized list
                    if person_id not in self.authorized_ids:
                        # Create event
                        event = {
                            'event_type': 'unauthorized_access',
                            'confidence': person['confidence_score'],
                            'bounding_box': person['bounding_box'],
                            'zone_id': entry_door_zone['id'],
                            'object_id': person_id
                        }
                        unauthorized_events.append(event)
                        
                        # Mark alert as sent
                        history['alert_sent'] = True
            else:
                # New person, initialize history
                self.person_history[person_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'positions': [person['bounding_box']],
                    'authorized': person_id in self.authorized_ids,
                    'alert_sent': False
                }
        
        # Clean up history for people not seen recently
        for person_id in list(self.person_history.keys()):
            history = self.person_history[person_id]
            if current_time - history['last_seen'] > 5.0:  # 5 seconds timeout
                del self.person_history[person_id]
                
        return unauthorized_events
    
    def detect_emergency_button_press(self, frame: np.ndarray, objects: List[Dict],
                                    emergency_button_zone: Dict) -> Optional[Dict]:
        """
        Detect if someone is pressing the emergency button.
        
        Args:
            frame: Input frame
            objects: Detected objects
            emergency_button_zone: Zone definition for emergency button
            
        Returns:
            Emergency button press event if detected, None otherwise
        """
        # This is a simplified implementation
        # In a real system, we would:
        # 1. Detect the actual button and its state (pressed/not pressed)
        # 2. Detect hand proximity to the button
        # 3. Use temporal information to detect the pressing action
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Filter people near emergency button
        people_near_button = []
        for obj in objects:
            if obj['type'] != 'person':
                continue
                
            # Get person bounding box
            person_bbox = obj['bounding_box']
            
            # Calculate person center
            person_center = (person_bbox[0] + person_bbox[2]/2, person_bbox[1] + person_bbox[3]/2)
            
            # Calculate emergency button center (approximating from polygon)
            button_points = np.array(emergency_button_zone['polygon'])
            button_center = np.mean(button_points, axis=0)
            
            # Calculate distance
            distance = np.sqrt((person_center[0] - button_center[0])**2 + 
                             (person_center[1] - button_center[1])**2)
            
            # Check if person is close enough to press the button
            if distance < 100:  # Threshold distance
                people_near_button.append({
                    'person': obj,
                    'distance': distance
                })
        
        # Check if anyone is close enough to press the button
        if people_near_button:
            # Sort by distance
            people_near_button.sort(key=lambda x: x['distance'])
            
            # Get closest person
            closest_person = people_near_button[0]['person']
            closest_distance = people_near_button[0]['distance']
            
            # Calculate proximity confidence (closer = higher confidence)
            proximity_confidence = max(0, (100 - closest_distance) / 100)
            
            # Check if the button is being pressed
            if self.emergency_button_status == 'normal':
                # Button was not pressed before, check if it's being pressed now
                if proximity_confidence > 0.7:  # Threshold for press detection
                    self.emergency_button_status = 'pressed'
                    self.button_press_time = current_time
                    self.button_press_confidence = proximity_confidence
                    
                    # Create button press event
                    event = {
                        'event_type': 'emergency_button_press',
                        'confidence': proximity_confidence,
                        'person_bbox': closest_person['bounding_box'],
                        'zone_id': emergency_button_zone['id'],
                        'object_id': closest_person.get('tracking_id', ''),
                        'timestamp': current_time
                    }
                    return event
            else:
                # Button was already pressed, check if it's still being pressed
                if proximity_confidence > 0.5:
                    # Still pressed, update confidence
                    self.button_press_confidence = proximity_confidence
                    return None  # Don't create a new event
                else:
                    # No longer pressed
                    self.emergency_button_status = 'normal'
                    self.button_press_time = None
                    self.button_press_confidence = 0.0
        elif self.emergency_button_status == 'pressed':
            # No one near button but it was pressed before, reset button status
            self.emergency_button_status = 'normal'
            self.button_press_time = None
            self.button_press_confidence = 0.0
            
        return None
    
    def process_frame(self, frame: np.ndarray, objects: List[Dict], zones: Dict) -> List[Dict]:
        """
        Process a frame for data center entry use cases.
        
        Args:
            frame: Input frame
            objects: Detected objects
            zones: Zone definitions
            
        Returns:
            List of detected events
        """
        events = []
        
        # Get zone definitions
        entry_door_zone = next((z for z in zones if z['id'] == 'dc_entry_door'), None)
        emergency_button_zone = next((z for z in zones if z['id'] == 'emergency_button'), None)
        
        # Run detectors if zones are defined
        if entry_door_zone:
            unauthorized_events = self.detect_unauthorized_access(frame, objects, entry_door_zone)
            events.extend(unauthorized_events)
            
        if emergency_button_zone:
            button_event = self.detect_emergency_button_press(frame, objects, emergency_button_zone)
            if button_event:
                events.append(button_event)
            
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
            
            if event_type == 'unauthorized_access':
                # Draw red box around unauthorized person
                bbox = event['bounding_box']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated, "ALERT: Unauthorized Access", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            elif event_type == 'emergency_button_press':
                # Draw red box around person pressing emergency button
                bbox = event['person_bbox']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated, "ALERT: Emergency Button Press", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Highlight emergency button area
                button_zone_id = event['zone_id']
                for zone in self.system.zones.get(button_zone_id, []):
                    if zone['id'] == button_zone_id:
                        # Draw button zone with red fill
                        points = np.array(zone['polygon'], np.int32)
                        points = points.reshape((-1, 1, 2))
                        cv2.fillPoly(annotated, [points], (0, 0, 255, 128))
                        break
                
        return annotated