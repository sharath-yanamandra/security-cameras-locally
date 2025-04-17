import cv2
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional
from core_system import DataCenterMonitoringSystem

class DataCenterInsideMonitor:
    """
    Specialized monitor for inside data center use cases:
    1. Person coming from unauthorized zone (e.g., fire service area)
    """
    
    def __init__(self, system: DataCenterMonitoringSystem):
        """
        Initialize the inside data center monitor.
        
        Args:
            system: The core monitoring system
        """
        self.system = system
        
        # Keep track of people's zone movements
        self.person_zone_history = {}
        
        # Authorized zones for different personnel types
        self.authorized_zone_map = {
            'staff': ['server_area', 'maintenance_path'],
            'maintenance': ['maintenance_path'],
            'security': ['server_area', 'maintenance_path', 'fire_exit'],
            'visitor': []  # Visitors not authorized in any zones
        }
        
        # Default personnel type (in a real system, this would be determined by recognition)
        self.default_personnel_type = 'visitor'
    
    def detect_unauthorized_zone_transitions(self, frame: np.ndarray, objects: List[Dict], 
                                          zones: List[Dict]) -> List[Dict]:
        """
        Detect people moving from unauthorized zones to sensitive areas.
        
        Args:
            frame: Input frame
            objects: Detected objects
            zones: List of zone definitions
            
        Returns:
            List of unauthorized zone transition events
        """
        transition_events = []
        
        # Create zone map for easy lookup
        zone_map = {zone['id']: zone for zone in zones}
        
        # Current timestamp
        current_time = time.time()
        
        # Process each person
        for obj in objects:
            if obj['type'] != 'person':
                continue
                
            person_id = obj.get('tracking_id', '')
            if not person_id:
                continue
                
            # Determine current zone
            current_zone = None
            for zone_id, zone in zone_map.items():
                if self.system._is_in_zone(obj['bounding_box'], zone):
                    current_zone = zone_id
                    break
                    
            if not current_zone:
                continue  # Person not in any defined zone
                
            # Get or initialize history
            if person_id in self.person_zone_history:
                history = self.person_zone_history[person_id]
                last_zone = history['zones'][-1] if history['zones'] else None
                
                # Check for zone transition
                if current_zone != last_zone:
                    # Person moved to a new zone
                    history['zones'].append(current_zone)
                    history['timestamps'].append(current_time)
                    
                    # Keep history to a reasonable size
                    if len(history['zones']) > 10:
                        history['zones'].pop(0)
                        history['timestamps'].pop(0)
                    
                    # Check if this transition is unauthorized
                    if last_zone and current_zone:
                        # Check specifically for transitions from fire exit to server area
                        if last_zone == 'fire_exit' and current_zone == 'server_area':
                            # Create unauthorized transition event
                            event = {
                                'event_type': 'unauthorized_zone_transition',
                                'confidence': obj['confidence_score'],
                                'bounding_box': obj['bounding_box'],
                                'from_zone': last_zone,
                                'to_zone': current_zone,
                                'object_id': person_id,
                                'personnel_type': history.get('personnel_type', self.default_personnel_type)
                            }
                            transition_events.append(event)
                            
                            # Mark as alerted
                            history['alerted'] = True
                
                # Update last seen time
                history['last_seen'] = current_time
            else:
                # Initialize history for new person
                self.person_zone_history[person_id] = {
                    'zones': [current_zone] if current_zone else [],
                    'timestamps': [current_time],
                    'personnel_type': self.default_personnel_type,  # Would be from recognition
                    'last_seen': current_time,
                    'alerted': False
                }
        
        # Clean up history for people not seen recently
        for person_id in list(self.person_zone_history.keys()):
            history = self.person_zone_history[person_id]
            if current_time - history['last_seen'] > 60.0:  # 1 minute timeout
                del self.person_zone_history[person_id]
                
        return transition_events
    
    def detect_zone_violations(self, frame: np.ndarray, objects: List[Dict], 
                            zones: List[Dict]) -> List[Dict]:
        """
        Detect people in zones they are not authorized to be in.
        
        Args:
            frame: Input frame
            objects: Detected objects
            zones: List of zone definitions
            
        Returns:
            List of zone violation events
        """
        violation_events = []
        
        # Create zone map for easy lookup
        zone_map = {zone['id']: zone for zone in zones}
        
        # Current timestamp
        current_time = time.time()
        
        # Process each person
        for obj in objects:
            if obj['type'] != 'person':
                continue
                
            person_id = obj.get('tracking_id', '')
            if not person_id:
                continue
                
            # Get personnel type from history or use default
            personnel_type = self.default_personnel_type
            if person_id in self.person_zone_history:
                personnel_type = self.person_zone_history[person_id].get('personnel_type', self.default_personnel_type)
            
            # Get authorized zones for this personnel type
            authorized_zones = self.authorized_zone_map.get(personnel_type, [])
            
            # Check each zone for violations
            for zone_id, zone in zone_map.items():
                if self.system._is_in_zone(obj['bounding_box'], zone):
                    # Person is in this zone
                    if zone_id not in authorized_zones:
                        # Create zone violation event
                        event = {
                            'event_type': 'zone_violation',
                            'confidence': obj['confidence_score'],
                            'bounding_box': obj['bounding_box'],
                            'zone_id': zone_id,
                            'object_id': person_id,
                            'personnel_type': personnel_type
                        }
                        violation_events.append(event)
                        
                        # Update history
                        if person_id in self.person_zone_history:
                            self.person_zone_history[person_id]['alerted'] = True
                    
                    break  # We found which zone the person is in
        
        return violation_events
    
    def process_frame(self, frame: np.ndarray, objects: List[Dict], zones: List[Dict]) -> List[Dict]:
        """
        Process a frame for inside data center use cases.
        
        Args:
            frame: Input frame
            objects: Detected objects
            zones: Zone definitions
            
        Returns:
            List of detected events
        """
        events = []
        
        # Detect unauthorized zone transitions
        transition_events = self.detect_unauthorized_zone_transitions(frame, objects, zones)
        events.extend(transition_events)
        
        # Detect zone violations
        violation_events = self.detect_zone_violations(frame, objects, zones)
        events.extend(violation_events)
        
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
            
            if event_type == 'unauthorized_zone_transition':
                # Draw red box around person making unauthorized transition
                bbox = event['bounding_box']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Show transition information
                from_zone = event.get('from_zone', 'unknown')
                to_zone = event.get('to_zone', 'unknown')
                cv2.putText(annotated, f"ALERT: Unauthorized {from_zone} -> {to_zone}", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            elif event_type == 'zone_violation':
                # Draw yellow box around person in unauthorized zone
                bbox = event['bounding_box']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
                # Show zone violation information
                zone_id = event.get('zone_id', 'unknown')
                personnel_type = event.get('personnel_type', 'unknown')
                cv2.putText(annotated, f"ALERT: {personnel_type} in {zone_id}", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
        return annotated