import os
import cv2
import time
import uuid
import json
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime
from filterpy.kalman import KalmanFilter
from typing import Dict, List, Tuple, Optional, Any

class DataCenterMonitoringSystem:
    def __init__(self, config_path: str):
        """
        Initialize the Data Center Monitoring System.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Skip database initialization - mock it
        self.db_conn = None
        print("Running in local mode without database connectivity")
        
        # Load AI models
        self.object_detector = self._load_object_detector()
        self.pose_estimator = self._load_pose_estimator()
        
        # Initialize trackers
        self.trackers = {}
        
        # Load zones configuration
        self.zones = self._load_zones()
        
        # Load rules
        self.rules = self._load_rules()
        
        # Initialize storage paths
        self._init_storage()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _init_database(self):
        """Mock database initialization."""
        print("Database connection mocked for local development")
        return None
    
    def _load_object_detector(self) -> YOLO:
        """Load YOLOv11-large object detection model."""
        model_path = self.config['models']['object_detector']
        print(f"Loading object detector from {model_path}")
        model = YOLO(model_path)
        return model
    
    def _load_pose_estimator(self) -> YOLO:
        """Load YOLO pose estimation model."""
        model_path = self.config['models']['pose_estimator']
        print(f"Loading pose estimator from {model_path}")
        model = YOLO(model_path)
        return model
    
    def _load_zones(self) -> Dict:
        """Load zone configurations."""
        zone_path = self.config['zones_config']
        print(f"Loading zones from {zone_path}")
        try:
            with open(zone_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Zones file not found: {zone_path}. Using empty zones.")
            return {}
    
    def _load_rules(self) -> Dict:
        """Load monitoring rules."""
        rules_path = self.config['rules_config']
        print(f"Loading rules from {rules_path}")
        try:
            with open(rules_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Rules file not found: {rules_path}. Using empty rules.")
            return []
    
    def _init_storage(self) -> None:
        """Initialize storage directories for snapshots and video clips."""
        os.makedirs(self.config['storage']['snapshots'], exist_ok=True)
        os.makedirs(self.config['storage']['video_clips'], exist_ok=True)
    
    def _init_tracker(self, object_id: str, bbox: List[float], class_id: int) -> None:
        """
        Initialize a Kalman Filter tracker for a new object.
        
        Args:
            object_id: Unique ID for the tracked object
            bbox: Bounding box [x, y, width, height]
            class_id: Class ID of the detected object
        """
        # Initialize Kalman filter with state [x, y, width, height, dx, dy, dw, dh]
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
            [0, 0, 0, 0, 0, 0, 0, 1],  # dh = dh
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(4) * 10
        
        # Process noise
        kf.Q = np.eye(8) * 0.1
        kf.Q[4:, 4:] *= 0.01
        
        # Initial state
        kf.x = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0]).reshape(8, 1)
        
        # Initial state covariance
        kf.P = np.eye(8) * 100
        
        self.trackers[object_id] = {
            'filter': kf,
            'class_id': class_id,
            'last_update': time.time(),
            'history': [bbox]
        }
    
    def _update_tracker(self, object_id: str, bbox: List[float]) -> List[float]:
        """
        Update a tracker with new detection.
        
        Args:
            object_id: ID of the tracked object
            bbox: New bounding box [x, y, width, height]
            
        Returns:
            Updated bounding box
        """
        if object_id not in self.trackers:
            return bbox
        
        tracker = self.trackers[object_id]
        kf = tracker['filter']
        
        # Predict
        kf.predict()
        
        # Update with measurement
        measurement = np.array([bbox[0], bbox[1], bbox[2], bbox[3]]).reshape(4, 1)
        kf.update(measurement)
        
        # Get updated state
        updated_bbox = kf.x[:4].flatten().tolist()
        
        # Update tracker metadata
        tracker['last_update'] = time.time()
        tracker['history'].append(updated_bbox)
        
        return updated_bbox
    
    def _match_detections(self, prev_objects: List[Dict], new_detections: List[Dict]) -> List[Dict]:
        """
        Match new detections with previously tracked objects using IoU.
        
        Args:
            prev_objects: Previously tracked objects
            new_detections: New detections from the object detector
            
        Returns:
            List of matched and new objects with tracking IDs
        """
        # If no previous objects, assign new IDs to all detections
        if not prev_objects:
            for detection in new_detections:
                detection['tracking_id'] = str(uuid.uuid4())
            return new_detections
        
        matched_objects = []
        unmatched_detections = list(new_detections)
        
        # For each previous object, find the best matching detection
        for prev_obj in prev_objects:
            best_iou = 0.5  # Minimum IoU threshold
            best_match = None
            best_idx = -1
            
            for i, detection in enumerate(unmatched_detections):
                # Calculate IoU between previous and current detection
                iou = self._calculate_iou(prev_obj['bounding_box'], detection['bounding_box'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = detection
                    best_idx = i
            
            if best_match:
                # Match found, update tracking ID
                best_match['tracking_id'] = prev_obj['tracking_id']
                matched_objects.append(best_match)
                unmatched_detections.pop(best_idx)
            else:
                # No match found, object may have disappeared
                # We could keep it for a few frames with tracking only
                prev_obj['confidence_score'] *= 0.8  # Reduce confidence
                matched_objects.append(prev_obj)
        
        # Assign new IDs to unmatched detections
        for detection in unmatched_detections:
            detection['tracking_id'] = str(uuid.uuid4())
            matched_objects.append(detection)
        
        return matched_objects
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union for two bounding boxes.
        
        Args:
            bbox1: First bounding box [x, y, width, height]
            bbox2: Second bounding box [x, y, width, height]
            
        Returns:
            IoU score between 0 and 1
        """
        # Convert format from [x, y, w, h] to [x1, y1, x2, y2]
        box1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
        box2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]
        
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area
    
    def _is_in_zone(self, bbox: List[float], zone: Dict) -> bool:
        """
        Check if an object is in a defined zone.
        
        Args:
            bbox: Bounding box [x, y, width, height]
            zone: Zone definition
            
        Returns:
            True if object is in zone, False otherwise
        """
        # Get center point of the bbox
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        
        # Simple polygon point-in-polygon check
        zone_polygon = zone['polygon']
        n = len(zone_polygon)
        inside = False
        
        p1x, p1y = zone_polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = zone_polygon[i % n]
            if center_y > min(p1y, p2y):
                if center_y <= max(p1y, p2y):
                    if center_x <= max(p1x, p2x):
                        if p1y != p2y:
                            x_intersect = (center_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or center_x <= x_intersect:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in the frame using YOLOv11-large.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected objects
        """
        results = self.object_detector(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                
                # Get class name
                class_name = result.names[class_id]
                
                # Convert to [x, y, width, height] format
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                
                # Create detection object
                detection = {
                    'type': self._get_object_type(class_name),
                    'subtype': class_name,
                    'bounding_box': bbox,
                    'confidence_score': float(confidence),
                    'class_id': class_id
                }
                
                detections.append(detection)
                
        return detections
    
    def _get_object_type(self, class_name: str) -> str:
        """Map class name to object type (person, vehicle, object)."""
        if class_name in ['person']:
            return 'person'
        elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
            return 'vehicle'
        else:
            return 'object'
    
    def _estimate_poses(self, frame: np.ndarray, person_detections: List[Dict]) -> List[Dict]:
        """
        Estimate poses for detected people using YOLO pose.
        
        Args:
            frame: Input frame
            person_detections: List of detected people
            
        Returns:
            List of person detections with pose information
        """
        if not person_detections:
            return []
        
        # Prepare input for pose estimator (only need to run on person detections)
        results = self.pose_estimator(frame, verbose=False)
        
        for i, detection in enumerate(person_detections):
            if i < len(results):
                result = results[i]
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy().tolist()
                    detection['metadata'] = {'keypoints': keypoints}
                    
                    # Determine pose based on keypoints (simplified)
                    detection['pose'] = self._determine_pose(keypoints)
            
        return person_detections
    
    def _determine_pose(self, keypoints: List[List[float]]) -> str:
        """
        Determine person's pose based on keypoints.
        
        Args:
            keypoints: List of keypoints from pose estimator
            
        Returns:
            Pose classification (sitting, standing, etc.)
        """
        # This is a simplified implementation
        # In a production system, you would use a more sophisticated approach
        
        if len(keypoints) < 17:  # Need at least all COCO keypoints
            return 'unknown'
        
        # Extract key body parts
        shoulders = keypoints[5:7]  # Left and right shoulder
        hips = keypoints[11:13]     # Left and right hip
        knees = keypoints[13:15]    # Left and right knee
        ankles = keypoints[15:17]   # Left and right ankle
        
        # Calculate vertical differences
        shoulder_hip_diff = abs(shoulders[0][1] - hips[0][1])
        hip_knee_diff = abs(hips[0][1] - knees[0][1])
        knee_ankle_diff = abs(knees[0][1] - ankles[0][1])
        
        # Simple sitting detection
        if hip_knee_diff < 0.2 * shoulder_hip_diff:
            return 'sitting'
        
        # Simple running detection based on posture
        if knee_ankle_diff > 0.8 * hip_knee_diff:
            return 'running'
        
        # Default to standing
        return 'standing'
    
    def _detect_events(self, frame: np.ndarray, objects: List[Dict], camera_id: str) -> List[Dict]:
    
        events = []
        
        # Get camera zones with proper error handling
        try:
            # First try to access as nested structure (like in zones.json)
            camera_zones = self.zones.get(camera_id, {}).get('zones', [])
        except (AttributeError, TypeError):
            # Fallback to direct access
            camera_zones = self.zones.get(camera_id, [])
        
        if not camera_zones:
            # No zones found, print debug info and return empty events
            print(f"No zones found for camera_id: {camera_id}")
            return events
        
        # Check each rule for the camera
        for rule in self.rules:
            if camera_id not in rule.get('camera_ids', []):
                continue
                
            # Filter objects based on rule requirements
            rule_objects = [obj for obj in objects if self._object_matches_rule(obj, rule)]
            
            # Skip if no matching objects
            if not rule_objects and rule.get('require_objects', True):
                continue
                
            # Check zone-specific rules
            for zone in camera_zones:
                # Ensure zone is a dictionary with an id
                if not isinstance(zone, dict):
                    print(f"Warning: Zone is not a dictionary: {zone}")
                    continue
                    
                if 'id' not in zone:
                    print(f"Warning: Zone has no 'id' field: {zone}")
                    continue
                
                zone_id = zone['id']
                
                if zone_id not in rule.get('zone_ids', []) and not rule.get('apply_all_zones', False):
                    continue
                    
                # Verify polygon exists in zone
                if 'polygon' not in zone:
                    print(f"Warning: Zone has no 'polygon' field: {zone}")
                    continue
                    
                # Count objects in zone
                try:
                    objects_in_zone = [obj for obj in rule_objects if self._is_in_zone(obj['bounding_box'], zone)]
                    count_in_zone = len(objects_in_zone)
                except Exception as e:
                    print(f"Error checking objects in zone: {e}")
                    continue
                
                # Check count rules
                if 'min_count' in rule and count_in_zone < rule['min_count']:
                    continue
                    
                if 'max_count' in rule and count_in_zone > rule['max_count']:
                    # Trigger event for exceeding max count
                    try:
                        event = self._create_event(rule, camera_id, zone_id, objects_in_zone, frame)
                        events.append(event)
                    except Exception as e:
                        print(f"Error creating max_count event: {e}")
                
                # Check pose-specific rules
                if 'pose_types' in rule:
                    try:
                        pose_objects = [obj for obj in objects_in_zone 
                                    if obj.get('pose') in rule['pose_types']]
                        
                        if pose_objects:
                            event = self._create_event(rule, camera_id, zone_id, pose_objects, frame)
                            events.append(event)
                    except Exception as e:
                        print(f"Error checking pose rules: {e}")
                
                # Check custom rules
                if 'rule_type' in rule:
                    try:
                        if rule['rule_type'] == 'unauthorized_access' and objects_in_zone:
                            event = self._create_event(rule, camera_id, zone_id, objects_in_zone, frame)
                            events.append(event)
                            
                        elif rule['rule_type'] == 'parking_violation' and objects_in_zone:
                            # Check if vehicles are in no-parking zone
                            vehicles = [obj for obj in objects_in_zone if obj['type'] == 'vehicle']
                            if vehicles:
                                event = self._create_event(rule, camera_id, zone_id, vehicles, frame)
                                events.append(event)
                    except Exception as e:
                        print(f"Error checking custom rules: {e}")
        
        return events
        
    '''def _detect_events(self, frame: np.ndarray, objects: List[Dict], camera_id: str) -> List[Dict]:
        """
        Detect events based on objects, poses, and rules.
        
        Args:
            frame: Current frame
            objects: Detected objects with tracking
            camera_id: ID of the camera
            
        Returns:
            List of detected events
        """
        events = []
        
        # Get camera zones
        camera_zones = self.zones.get(camera_id, [])
        
        # Check each rule for the camera
        for rule in self.rules:
            if camera_id not in rule.get('camera_ids', []):
                continue
                
            # Filter objects based on rule requirements
            rule_objects = [obj for obj in objects if self._object_matches_rule(obj, rule)]
            
            # Skip if no matching objects
            if not rule_objects and rule.get('require_objects', True):
                continue
                
            # Check zone-specific rules
            for zone in camera_zones:
                zone_id = zone['id']
                
                if zone_id not in rule.get('zone_ids', []):
                    continue
                    
                # Count objects in zone
                objects_in_zone = [obj for obj in rule_objects if self._is_in_zone(obj['bounding_box'], zone)]
                count_in_zone = len(objects_in_zone)
                
                # Check count rules
                if 'min_count' in rule and count_in_zone < rule['min_count']:
                    continue
                    
                if 'max_count' in rule and count_in_zone > rule['max_count']:
                    # Trigger event for exceeding max count
                    event = self._create_event(rule, camera_id, zone_id, objects_in_zone, frame)
                    events.append(event)
                    
                # Check pose-specific rules
                if 'pose_types' in rule:
                    pose_objects = [obj for obj in objects_in_zone 
                                   if obj.get('pose') in rule['pose_types']]
                    
                    if pose_objects:
                        event = self._create_event(rule, camera_id, zone_id, pose_objects, frame)
                        events.append(event)
                
                # Check custom rules
                if 'rule_type' in rule:
                    if rule['rule_type'] == 'unauthorized_access' and objects_in_zone:
                        event = self._create_event(rule, camera_id, zone_id, objects_in_zone, frame)
                        events.append(event)
                        
                    elif rule['rule_type'] == 'parking_violation' and objects_in_zone:
                        # Check if vehicles are in no-parking zone
                        vehicles = [obj for obj in objects_in_zone if obj['type'] == 'vehicle']
                        if vehicles:
                            event = self._create_event(rule, camera_id, zone_id, vehicles, frame)
                            events.append(event)
        
        return events'''
    
    def _object_matches_rule(self, obj: Dict, rule: Dict) -> bool:
        """Check if an object matches the rule criteria."""
        if 'object_types' in rule and obj['type'] not in rule['object_types']:
            return False
            
        if 'object_subtypes' in rule and obj['subtype'] not in rule['object_subtypes']:
            return False
            
        return True
    
    def _create_event(self, rule: Dict, camera_id: str, zone_id: str, 
                     objects: List[Dict], frame: np.ndarray) -> Dict:
        """Create an event based on rule violation."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Calculate average confidence score
        avg_confidence = sum(obj['confidence_score'] for obj in objects) / len(objects) if objects else 0
        
        # Save snapshot
        snapshot_path = self._save_snapshot(event_id, frame)
        
        # Create event
        event = {
            'event_id': event_id,
            'rule_id': rule.get('id', 'unknown_rule'),
            'camera_id': camera_id,
            'zone_id': zone_id,
            'timestamp': timestamp,
            'confidence_score': avg_confidence,
            'metadata': {
                'object_count': len(objects),
                'rule_type': rule.get('rule_type', 'default'),
                'rule_name': rule.get('name', 'Unnamed Rule')
            },
            'snapshot_url': snapshot_path,
            'video_clip_url': '',  # Will be updated when clip is ready
            'status': 'new',
            'resolution_notes': '',
            'resolved_by': None,
            'resolved_at': None,
            'objects': objects
        }
        
        return event
    
    def _save_snapshot(self, event_id: str, frame: np.ndarray) -> str:
        """Save snapshot image for the event."""
        snapshot_dir = self.config['storage']['snapshots']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{event_id}_{timestamp}.jpg'
        filepath = os.path.join(snapshot_dir, filename)
        
        cv2.imwrite(filepath, frame)
        
        return filepath
    
    def _save_events_to_db(self, events: List[Dict]) -> None:
        """
        Mock saving events to the database - just print them.
        
        Args:
            events: List of events to save
        """
        if not events:
            return
            
        print(f"Detected {len(events)} events:")
        for event in events:
            rule_name = event['metadata'].get('rule_name', 'Unnamed rule')
            confidence = event['confidence_score']
            timestamp = event['timestamp']
            print(f"  - {rule_name} (confidence: {confidence:.2f}, time: {timestamp})")
            
            # Log object details
            for i, obj in enumerate(event['objects']):
                obj_type = obj['type']
                obj_subtype = obj['subtype']
                pose = obj.get('pose', 'unknown')
                print(f"     Object {i+1}: {obj_subtype} ({pose})")
    
    def process_frame(self, frame: np.ndarray, camera_id: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame from a camera.
        
        Args:
            frame: Input frame
            camera_id: ID of the camera
            
        Returns:
            Annotated frame and list of detected events
        """
        # Detect objects
        detections = self._detect_objects(frame)
        
        # Match with previous detections and update trackers
        tracked_objects = self._match_detections([], detections)  # TODO: use previous frame detections
        
        # Update Kalman filters for tracking
        for obj in tracked_objects:
            tracking_id = obj['tracking_id']
            bbox = obj['bounding_box']
            class_id = obj['class_id']
            
            if tracking_id not in self.trackers:
                self._init_tracker(tracking_id, bbox, class_id)
            else:
                updated_bbox = self._update_tracker(tracking_id, bbox)
                obj['bounding_box'] = updated_bbox
        
        # Extract person detections for pose estimation
        person_detections = [obj for obj in tracked_objects if obj['type'] == 'person']
        
        # Estimate poses for people
        person_detections = self._estimate_poses(frame, person_detections)
        
        # Update tracked_objects with pose information
        person_tracking_ids = [p['tracking_id'] for p in person_detections]
        tracked_objects = [obj if obj['tracking_id'] not in person_tracking_ids else 
                         next(p for p in person_detections if p['tracking_id'] == obj['tracking_id']) 
                         for obj in tracked_objects]
        
        # Detect events
        events = self._detect_events(frame, tracked_objects, camera_id)
        
        # Print events instead of saving to database
        if events:
            self._save_events_to_db(events)
        
        # Draw annotations
        annotated_frame = self._draw_annotations(frame, tracked_objects, events)
        
        return annotated_frame, tracked_objects
    
    def _draw_annotations(self, frame: np.ndarray, 
                         objects: List[Dict], events: List[Dict]) -> np.ndarray:
        """Draw annotations on the frame."""
        annotated = frame.copy()
        
        # Draw objects
        for obj in objects:
            bbox = obj['bounding_box']
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Different colors for different object types
            if obj['type'] == 'person':
                color = (0, 255, 0)  # Green for people
            elif obj['type'] == 'vehicle':
                color = (0, 0, 255)  # Red for vehicles
            else:
                color = (255, 0, 0)  # Blue for other objects
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{obj['subtype']} ({obj.get('pose', '')})"
            confidence = f"{obj['confidence_score']:.2f}"
            label_text = f"{label} {confidence}"
            
            cv2.putText(annotated, label_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Highlight events
        for event in events:
            # Draw a red background for event areas
            for obj in event['objects']:
                bbox = obj['bounding_box']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Create semi-transparent overlay
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
                
                # Add overlay
                cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
                
                # Add event text
                rule_name = event['metadata'].get('rule_name', 'Rule violation')
                cv2.putText(annotated, f"ALERT: {rule_name}", (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated
    
    def process_video(self, video_path: str, camera_id: str, output_path: Optional[str] = None) -> None:
        """
        Process a video file.
        
        Args:
            video_path: Path to the input video
            camera_id: ID of the camera
            output_path: Optional path for the output video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create output video writer if output_path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_events = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every nth frame for efficiency
            if frame_count % self.config.get('frame_skip', 1) == 0:
                try:
                    # Process frame
                    annotated_frame, objects = self.process_frame(frame, camera_id)
                    
                    # Write to output video
                    if output_path:
                        out.write(annotated_frame)
                        
                    # Display if needed
                    if self.config.get('display', True):  # Changed default to True for local testing
                        cv2.imshow('Processing', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Write original frame to output if we're skipping processing
                if output_path:
                    out.write(frame)
            
            # Show progress every 10 frames
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count} frames ({fps_processing:.2f} fps)")
        
        # Clean up
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Completed processing {frame_count} frames")
    
    def run_live_camera(self, camera_url: str, camera_id: str) -> None:
        """
        Process a live camera stream.
        
        Args:
            camera_url: URL or ID of the camera
            camera_id: ID of the camera in the system
        """
        cap = cv2.VideoCapture(camera_url)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_url}")
            return
            
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            frame_count += 1
            
            # Process every nth frame for efficiency
            if frame_count % self.config.get('frame_skip', 1) == 0:
                try:
                    # Process frame
                    annotated_frame, objects = self.process_frame(frame, camera_id)
                    
                    # Display the results
                    cv2.imshow(f'Camera {camera_id}', annotated_frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Show processing rate every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"Processed {frame_count} frames ({fps_processing:.2f} fps)")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


# Simplified version of EventRuleEngine that doesn't require database
class EventRuleEngine:
    """Rule engine for data center security events"""
    
    def __init__(self, config_path: str):
        """Initialize the rule engine with configuration."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            self.rules = self._load_rules()
        except Exception as e:
            print(f"Error initializing rule engine: {e}")
            self.config = {}
            self.rules = []
    
    def _load_rules(self) -> List[Dict]:
        """Load the rules from configuration."""
        rules_path = self.config.get('rules_file', 'rules.json')
        try:
            with open(rules_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Rules file not found: {rules_path}, using empty rules")
            return []
    
    def evaluate(self, camera_id: str, zone_id: str, objects: List[Dict]) -> List[Dict]:
        """
        Evaluate all rules against the current objects.
        
        Args:
            camera_id: ID of the camera
            zone_id: ID of the zone
            objects: List of detected objects
            
        Returns:
            List of triggered rules with event information
        """
        triggered_events = []
        
        for rule in self.rules:
            # Skip if rule doesn't apply to this camera/zone
            if camera_id not in rule.get('camera_ids', []):
                continue
                
            if zone_id not in rule.get('zone_ids', []) and not rule.get('apply_all_zones', False):
                continue
            
            # Check the rule condition
            if self._evaluate_rule(rule, objects):
                event = {
                    'rule_id': rule.get('id', 'unknown'),
                    'rule_name': rule.get('name', 'Unnamed Rule'),
                    'severity': rule.get('severity', 'medium'),
                    'objects': objects,
                    'timestamp': datetime.now().isoformat()
                }
                triggered_events.append(event)
        
        return triggered_events
    
    def _evaluate_rule(self, rule: Dict, objects: List[Dict]) -> bool:
        """
        Evaluate a single rule against the objects.
        
        Args:
            rule: Rule definition
            objects: List of detected objects
            
        Returns:
            True if rule is triggered, False otherwise
        """
        rule_type = rule.get('type')
        
        if rule_type == 'count':
            # Count objects of specified types
            count = 0
            for obj in objects:
                if obj['type'] in rule.get('object_types', []) and \
                   obj['confidence_score'] >= rule.get('min_confidence', 0.0):
                    count += 1
            
            min_count = rule.get('min_count')
            max_count = rule.get('max_count')
            
            if min_count is not None and count < min_count:
                return False
                
            if max_count is not None and count > max_count:
                return True
                
            return False
            
        elif rule_type == 'pose':
            # Check for specific poses
            required_poses = rule.get('poses', [])
            
            for obj in objects:
                if obj['type'] == 'person' and \
                   obj.get('pose') in required_poses and \
                   obj['confidence_score'] >= rule.get('min_confidence', 0.0):
                    return True
                    
            return False
            
        elif rule_type == 'proximity':
            # Check proximity between objects
            target_types = rule.get('target_types', [])
            reference_types = rule.get('reference_types', [])
            max_distance = rule.get('max_distance', 100)
            
            target_objects = [obj for obj in objects if obj['type'] in target_types]
            reference_objects = [obj for obj in objects if obj['type'] in reference_types]
            
            for target in target_objects:
                target_center = (target['bounding_box'][0] + target['bounding_box'][2]/2,
                                target['bounding_box'][1] + target['bounding_box'][3]/2)
                
                for reference in reference_objects:
                    ref_center = (reference['bounding_box'][0] + reference['bounding_box'][2]/2,
                                 reference['bounding_box'][1] + reference['bounding_box'][3]/2)
                    
                    # Calculate distance
                    distance = ((target_center[0] - ref_center[0])**2 + 
                               (target_center[1] - ref_center[1])**2)**0.5
                    
                    if distance <= max_distance:
                        return True
            
            return False
            
        # Default case
        return False


def main():
    """Main entry point for direct testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Center Monitoring System - Local Mode')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['video', 'live'], required=True,
                       help='Processing mode: video or live camera')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to video file or camera URL/ID')
    parser.add_argument('--camera_id', type=str, required=True,
                       help='Camera ID in the system')
    parser.add_argument('--output', type=str, help='Path to output video (video mode only)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = DataCenterMonitoringSystem(args.config)
    
    # Process based on mode
    if args.mode == 'video':
        system.process_video(args.input, args.camera_id, args.output)
    else:  # Live mode
        system.run_live_camera(args.input, args.camera_id)


if __name__ == "__main__":
    main()