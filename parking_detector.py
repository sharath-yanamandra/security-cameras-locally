import cv2
import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional
from core_system import DataCenterMonitoringSystem

class ParkingAreaMonitor:
    """
    Specialized monitor for parking area use cases:
    1. Vehicles parked outside of designated zones
    2. Vehicles blocking exits
    3. Suspicious activities in parking area
    """
    
    def __init__(self, system: DataCenterMonitoringSystem):
        """
        Initialize the parking area monitor.
        
        Args:
            system: The core monitoring system
        """
        self.system = system
        
        # Vehicle tracking history for parking duration
        self.vehicle_history = {}
        
        # Person tracking for suspicious activity
        self.person_history = {}
        
        # Constants
        self.suspicious_speed_threshold = 150  # Pixels per second for running
        self.min_parking_time = 30  # Seconds to consider a vehicle "parked"
    
    def detect_improper_parking(self, frame: np.ndarray, objects: List[Dict], 
                             parking_zones: Dict, no_parking_zones: Dict) -> List[Dict]:
        """
        Detect vehicles parked in no-parking zones.
        
        Args:
            frame: Input frame
            objects: Detected objects
            parking_zones: Zone definition for allowed parking
            no_parking_zones: Zone definition for no-parking areas
            
        Returns:
            List of improper parking events
        """
        parking_events = []
        
        # Current timestamp
        current_time = time.time()
        
        # Get all vehicles in the frame
        vehicles = [obj for obj in objects if obj['type'] == 'vehicle']
        
        for vehicle in vehicles:
            vehicle_id = vehicle.get('tracking_id', '')
            if not vehicle_id:
                continue
                
            # Check if vehicle is in no-parking zone
            in_no_parking_zone = any(self.system._is_in_zone(vehicle['bounding_box'], zone) 
                                  for zone in no_parking_zones)
                                  
            # Check if vehicle is outside all parking zones
            in_parking_zone = any(self.system._is_in_zone(vehicle['bounding_box'], zone) 
                               for zone in parking_zones)
            
            # Get or create vehicle history
            if vehicle_id in self.vehicle_history:
                vehicle_data = self.vehicle_history[vehicle_id]
                
                # Update position
                bbox = vehicle['bounding_box']
                center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                vehicle_data['positions'].append(center)
                vehicle_data['timestamps'].append(current_time)
                
                # Keep history at a reasonable size
                if len(vehicle_data['positions']) > 30:
                    vehicle_data['positions'].pop(0)
                    vehicle_data['timestamps'].pop(0)
                
                # Calculate time in current zone
                time_in_position = current_time - vehicle_data.get('zone_entry_time', current_time)
                
                # Check if vehicle has been in a no-parking zone for a while
                if in_no_parking_zone:
                    if vehicle_data.get('in_no_parking_zone', False):
                        # Vehicle was already in no-parking zone
                        if time_in_position > self.min_parking_time and not vehicle_data.get('no_parking_alerted', False):
                            # Create parking violation event
                            event = {
                                'event_type': 'no_parking_violation',
                                'confidence': vehicle['confidence_score'],
                                'bounding_box': vehicle['bounding_box'],
                                'vehicle_type': vehicle['subtype'],
                                'duration': time_in_position,
                                'object_id': vehicle_id
                            }
                            parking_events.append(event)
                            
                            # Mark as alerted
                            vehicle_data['no_parking_alerted'] = True
                    else:
                        # Vehicle just entered no-parking zone
                        vehicle_data['in_no_parking_zone'] = True
                        vehicle_data['zone_entry_time'] = current_time
                        vehicle_data['no_parking_alerted'] = False
                else:
                    # Reset no-parking flags if vehicle left the zone
                    vehicle_data['in_no_parking_zone'] = False
                    vehicle_data['no_parking_alerted'] = False
                
                # Check if vehicle is outside all parking zones but still in frame
                if not in_parking_zone and not in_no_parking_zone:
                    if vehicle_data.get('outside_parking_zone', False):
                        # Vehicle was already outside parking zone
                        if time_in_position > self.min_parking_time and not vehicle_data.get('outside_parking_alerted', False):
                            # Create parking violation event
                            event = {
                                'event_type': 'outside_parking_violation',
                                'confidence': vehicle['confidence_score'],
                                'bounding_box': vehicle['bounding_box'],
                                'vehicle_type': vehicle['subtype'],
                                'duration': time_in_position,
                                'object_id': vehicle_id
                            }
                            parking_events.append(event)
                            
                            # Mark as alerted
                            vehicle_data['outside_parking_alerted'] = True
                    else:
                        # Vehicle just went outside parking zone
                        vehicle_data['outside_parking_zone'] = True
                        vehicle_data['zone_entry_time'] = current_time
                        vehicle_data['outside_parking_alerted'] = False
                else:
                    # Reset outside parking flags if vehicle entered a defined zone
                    vehicle_data['outside_parking_zone'] = False
                    vehicle_data['outside_parking_alerted'] = False
                
                # Update last seen time
                vehicle_data['last_seen'] = current_time
            else:
                # Initialize history for new vehicle
                self.vehicle_history[vehicle_id] = {
                    'positions': [self._get_center(vehicle['bounding_box'])],
                    'timestamps': [current_time],
                    'in_no_parking_zone': in_no_parking_zone,
                    'outside_parking_zone': not in_parking_zone and not in_no_parking_zone,
                    'zone_entry_time': current_time,
                    'no_parking_alerted': False,
                    'outside_parking_alerted': False,
                    'last_seen': current_time
                }
        
        # Clean up history for vehicles not seen recently
        for vehicle_id in list(self.vehicle_history.keys()):
            if current_time - self.vehicle_history[vehicle_id]['last_seen'] > 60.0:  # 1 minute timeout
                del self.vehicle_history[vehicle_id]
                
        return parking_events
    
    def detect_exit_blocking(self, frame: np.ndarray, objects: List[Dict], 
                          exit_zones: List[Dict]) -> List[Dict]:
        """
        Detect vehicles blocking exit paths.
        
        Args:
            frame: Input frame
            objects: Detected objects
            exit_zones: Zone definitions for exit paths
            
        Returns:
            List of exit blocking events
        """
        blocking_events = []
        
        # Get all vehicles in the frame
        vehicles = [obj for obj in objects if obj['type'] == 'vehicle']
        
        for exit_zone in exit_zones:
            # Find vehicles in this exit zone
            vehicles_in_exit = [v for v in vehicles if self.system._is_in_zone(v['bounding_box'], exit_zone)]
            
            if vehicles_in_exit:
                # Exit is blocked
                for vehicle in vehicles_in_exit:
                    # Create exit blocking event
                    event = {
                        'event_type': 'exit_blocking',
                        'confidence': vehicle['confidence_score'],
                        'bounding_box': vehicle['bounding_box'],
                        'vehicle_type': vehicle['subtype'],
                        'zone_id': exit_zone['id'],
                        'object_id': vehicle.get('tracking_id', '')
                    }
                    blocking_events.append(event)
        
        return blocking_events
    
    def detect_suspicious_activity(self, frame: np.ndarray, objects: List[Dict]) -> List[Dict]:
        """
        Detect suspicious activities in parking area.
        
        Args:
            frame: Input frame
            objects: Detected objects
            
        Returns:
            List of suspicious activity events
        """
        suspicious_events = []
        
        # Current timestamp
        current_time = time.time()
        
        # Get all people in the frame
        people = [obj for obj in objects if obj['type'] == 'person']
        
        for person in people:
            person_id = person.get('tracking_id', '')
            if not person_id:
                continue
                
            # Check for suspicious poses directly
            pose = person.get('pose', '')
            if pose in ['crouching', 'running']:
                # Create suspicious pose event
                event = {
                    'event_type': 'suspicious_pose',
                    'confidence': person['confidence_score'],
                    'bounding_box': person['bounding_box'],
                    'pose': pose,
                    'object_id': person_id
                }
                suspicious_events.append(event)
            
            # Get or create person history for velocity-based detection
            bbox = person['bounding_box']
            center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            
            if person_id in self.person_history:
                person_data = self.person_history[person_id]
                
                # Update position
                person_data['positions'].append(center)
                person_data['timestamps'].append(current_time)
                
                # Keep history at a reasonable size
                if len(person_data['positions']) > 10:
                    person_data['positions'].pop(0)
                    person_data['timestamps'].pop(0)
                
                # Calculate velocity if we have enough history
                if len(person_data['positions']) >= 3:
                    # Calculate displacement
                    pos1 = person_data['positions'][0]
                    pos2 = person_data['positions'][-1]
                    time1 = person_data['timestamps'][0]
                    time2 = person_data['timestamps'][-1]
                    
                    dx = pos2[0] - pos1[0]
                    dy = pos2[1] - pos1[1]
                    dt = time2 - time1
                    
                    # Calculate speed in pixels per second
                    if dt > 0:
                        speed = np.sqrt(dx*dx + dy*dy) / dt
                        
                        # Check if person is moving very fast (running)
                        if speed > self.suspicious_speed_threshold and not person_data.get('speed_alerted', False):
                            # Create suspicious speed event
                            event = {
                                'event_type': 'suspicious_speed',
                                'confidence': person['confidence_score'],
                                'bounding_box': person['bounding_box'],
                                'speed': speed,
                                'object_id': person_id
                            }
                            suspicious_events.append(event)
                            
                            # Mark as alerted
                            person_data['speed_alerted'] = True
                
                # Update last seen time
                person_data['last_seen'] = current_time
            else:
                # Initialize history for new person
                self.person_history[person_id] = {
                    'positions': [center],
                    'timestamps': [current_time],
                    'speed_alerted': False,
                    'last_seen': current_time
                }
        
        # Clean up history for people not seen recently
        for person_id in list(self.person_history.keys()):
            if current_time - self.person_history[person_id]['last_seen'] > 30.0:  # 30 seconds timeout
                del self.person_history[person_id]
                
        return suspicious_events
    
    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of a bounding box."""
        return (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
    
    def process_frame(self, frame: np.ndarray, objects: List[Dict], zones: Dict) -> List[Dict]:
        """
        Process a frame for parking area use cases.
        
        Args:
            frame: Input frame
            objects: Detected objects
            zones: Zone definitions
            
        Returns:
            List of detected events
        """
        events = []
        
        # Get zone definitions
        parking_zones = [z for z in zones if z.get('type') == 'parking_zone']
        no_parking_zones = [z for z in zones if z.get('type') == 'no_parking_zone']
        exit_zones = [z for z in zones if z.get('type') == 'exit_zone']
        
        # Run detectors if zones are defined
        if parking_zones and no_parking_zones:
            parking_events = self.detect_improper_parking(frame, objects, parking_zones, no_parking_zones)
            events.extend(parking_events)
            
        if exit_zones:
            blocking_events = self.detect_exit_blocking(frame, objects, exit_zones)
            events.extend(blocking_events)
        
        # Suspicious activity detection works in all zones
        suspicious_events = self.detect_suspicious_activity(frame, objects)
        events.extend(suspicious_events)
        
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
            bbox = event.get('bounding_box')
            
            if not bbox:
                continue
                
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            if event_type == 'no_parking_violation':
                # Draw red box around vehicle in no-parking zone
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                duration = int(event.get('duration', 0))
                cv2.putText(annotated, f"ALERT: No Parking Violation ({duration}s)", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            elif event_type == 'outside_parking_violation':
                # Draw orange box around vehicle outside parking zone
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)
                duration = int(event.get('duration', 0))
                cv2.putText(annotated, f"ALERT: Improper Parking ({duration}s)", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
            elif event_type == 'exit_blocking':
                # Draw red box around vehicle blocking exit
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated, "ALERT: Exit Blocked", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            elif event_type == 'suspicious_pose':
                # Draw yellow box around person with suspicious pose
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
                pose = event.get('pose', 'unknown')
                cv2.putText(annotated, f"ALERT: Suspicious Activity ({pose})", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
            elif event_type == 'suspicious_speed':
                # Draw yellow box around person moving suspiciously fast
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
                speed = int(event.get('speed', 0))
                cv2.putText(annotated, f"ALERT: Running Person ({speed} px/s)", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
        return annotated