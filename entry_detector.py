import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from core_system import DataCenterMonitoringSystem

class EntryAreaMonitor:
    """
    Specialized monitor for entry area use cases:
    1. Sitting on stairs detection
    2. Gathering detection
    3. Vehicle entry and stopping detection
    """
    
    def __init__(self, system: DataCenterMonitoringSystem):
        """
        Initialize the entry area monitor.
        
        Args:
            system: The core monitoring system
        """
        self.system = system
        self.sitting_detector = self._init_sitting_detector()
        self.gathering_detector = self._init_gathering_detector()
        self.vehicle_detector = self._init_vehicle_detector()
        
        # History for tracking objects over time
        self.person_history = {}
        self.vehicle_history = {}
        
    def _init_sitting_detector(self):
        """Initialize pose classifier for sitting detection."""
        # For this implementation, we'll use the pose estimator from the core system
        return None
        
    def _init_gathering_detector(self):
        """Initialize group gathering detector."""
        # Simple implementation based on counting and proximity
        return None
        
    def _init_vehicle_detector(self):
        """Initialize vehicle detection and tracking."""
        # Use the object detector from the core system
        return None
    
    def detect_sitting_on_stairs(self, frame: np.ndarray, objects: List[Dict], 
                                stairs_zone: Dict) -> List[Dict]:
        """
        Detect people sitting on stairs.
        
        Args:
            frame: Input frame
            objects: Detected objects
            stairs_zone: Zone definition for stairs
            
        Returns:
            List of detected sitting events
        """
        sitting_events = []
        
        # Filter people in stairs zone
        people_in_stairs = [obj for obj in objects 
                          if obj['type'] == 'person' and 
                          self.system._is_in_zone(obj['bounding_box'], stairs_zone)]
        
        # Check for sitting pose
        for person in people_in_stairs:
            if person.get('pose') == 'sitting':
                # Create sitting event
                event = {
                    'event_type': 'sitting_on_stairs',
                    'confidence': person['confidence_score'],
                    'bounding_box': person['bounding_box'],
                    'zone_id': stairs_zone['id'],
                    'object_id': person.get('tracking_id', '')
                }
                sitting_events.append(event)
                
        return sitting_events
    
    def detect_gathering(self, frame: np.ndarray, objects: List[Dict], 
                       entry_zone: Dict) -> Optional[Dict]:
        """
        Detect gathering of people at entry.
        
        Args:
            frame: Input frame
            objects: Detected objects
            entry_zone: Zone definition for entry
            
        Returns:
            Gathering event if detected, None otherwise
        """
        # Filter people in entry zone
        people_in_entry = [obj for obj in objects 
                          if obj['type'] == 'person' and 
                          self.system._is_in_zone(obj['bounding_box'], entry_zone)]
        
        # Check if the number of people exceeds the threshold
        if len(people_in_entry) > 5:  # Threshold for gathering
            # Create gathering event
            event = {
                'event_type': 'gathering_at_entry',
                'person_count': len(people_in_entry),
                'confidence': sum(p['confidence_score'] for p in people_in_entry) / len(people_in_entry),
                'zone_id': entry_zone['id']
            }
            return event
            
        return None
    
    def detect_vehicle_entry(self, frame: np.ndarray, objects: List[Dict], 
                           vehicle_zone: Dict) -> List[Dict]:
        """
        Detect vehicles stopping in entry zone.
        
        Args:
            frame: Input frame
            objects: Detected objects
            vehicle_zone: Zone definition for vehicle entry
            
        Returns:
            List of detected vehicle entry events
        """
        vehicle_events = []
        
        # Filter vehicles in entry zone
        vehicles_in_entry = [obj for obj in objects 
                           if obj['type'] == 'vehicle' and 
                           self.system._is_in_zone(obj['bounding_box'], vehicle_zone)]
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Update vehicle history and check for stopped vehicles
        for vehicle in vehicles_in_entry:
            vehicle_id = vehicle.get('tracking_id', '')
            
            if not vehicle_id:
                continue
                
            bbox = vehicle['bounding_box']
            center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            
            if vehicle_id in self.vehicle_history:
                # Get last position
                history = self.vehicle_history[vehicle_id]
                last_center = history['positions'][-1]
                last_time = history['timestamps'][-1]
                
                # Calculate movement
                distance = ((center[0] - last_center[0])**2 + 
                          (center[1] - last_center[1])**2)**0.5
                time_diff = current_time - last_time
                
                # Update history
                history['positions'].append(center)
                history['timestamps'].append(current_time)
                history['distances'].append(distance)
                
                # Keep history to a reasonable size
                if len(history['positions']) > 30:  # About 5 seconds at 6 fps
                    history['positions'].pop(0)
                    history['timestamps'].pop(0)
                    history['distances'].pop(0)
                
                # Check if vehicle is stopped (very small movement over time)
                if len(history['distances']) > 10:  # Need enough history
                    avg_distance = sum(history['distances']) / len(history['distances'])
                    if avg_distance < 5:  # Threshold for "stopped"
                        # Calculate how long the vehicle has been stopped
                        stop_duration = current_time - history.get('stop_start_time', current_time)
                        
                        if not history.get('stop_detected', False):
                            # Mark the start of stopping
                            history['stop_start_time'] = current_time
                            history['stop_detected'] = True
                        elif stop_duration > 30:  # Stopped for more than 30 seconds
                            # Create vehicle stopped event
                            event = {
                                'event_type': 'vehicle_stopped',
                                'vehicle_type': vehicle['subtype'],
                                'confidence': vehicle['confidence_score'],
                                'bounding_box': vehicle['bounding_box'],
                                'zone_id': vehicle_zone['id'],
                                'object_id': vehicle_id,
                                'duration': stop_duration
                            }
                            vehicle_events.append(event)
                    else:
                        # Vehicle is moving, reset stop detection
                        history['stop_detected'] = False
            else:
                # New vehicle, initialize history
                self.vehicle_history[vehicle_id] = {
                    'positions': [center],
                    'timestamps': [current_time],
                    'distances': [],
                    'stop_detected': False
                }
        
        # Clean up history for vehicles that are no longer in the frame
        current_vehicle_ids = [v.get('tracking_id', '') for v in vehicles_in_entry]
        for vehicle_id in list(self.vehicle_history.keys()):
            if vehicle_id not in current_vehicle_ids:
                # Vehicle left the frame
                del self.vehicle_history[vehicle_id]
                
        return vehicle_events
    
    def process_frame(self, frame: np.ndarray, objects: List[Dict], zones: Dict) -> List[Dict]:
        """
        Process a frame for entry area use cases.
        
        Args:
            frame: Input frame
            objects: Detected objects
            zones: Zone definitions
            
        Returns:
            List of detected events
        """
        events = []
        
        # Get zone definitions
        stairs_zone = next((z for z in zones if z['id'] == 'entry_stairs'), None)
        entry_zone = next((z for z in zones if z['id'] == 'entry_gate'), None)
        vehicle_zone = next((z for z in zones if z['id'] == 'vehicle_entry'), None)
        
        # Run detectors if zones are defined
        if stairs_zone:
            sitting_events = self.detect_sitting_on_stairs(frame, objects, stairs_zone)
            events.extend(sitting_events)
            
        if entry_zone:
            gathering_event = self.detect_gathering(frame, objects, entry_zone)
            if gathering_event:
                events.append(gathering_event)
                
        if vehicle_zone:
            vehicle_events = self.detect_vehicle_entry(frame, objects, vehicle_zone)
            events.extend(vehicle_events)
            
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
            
            if event_type == 'sitting_on_stairs':
                # Draw red box around person sitting on stairs
                bbox = event['bounding_box']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated, "ALERT: Sitting on Stairs", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            elif event_type == 'gathering_at_entry':
                # Draw warning text for gathering
                cv2.putText(annotated, f"ALERT: Gathering ({event['person_count']} people)",
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            elif event_type == 'vehicle_stopped':
                # Draw red box around stopped vehicle
                bbox = event['bounding_box']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                duration = int(event.get('duration', 0))
                cv2.putText(annotated, f"ALERT: Stopped Vehicle ({duration}s)",
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        return annotated