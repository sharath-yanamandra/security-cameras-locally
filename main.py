#!/usr/bin/env python
import os
import cv2
import argparse
import json
import time
from typing import Dict, List

# Import modified core system for local use
# Assumes you've saved the modified core system as core_system_local.py
from core_system_test import DataCenterMonitoringSystem

# Import specialized use case monitors
from entry_detector import EntryAreaMonitor
from reception_detector import ReceptionAreaMonitor
from datacenter_entry_detector import DataCenterEntryMonitor
from datacenter_inside_detector import DataCenterInsideMonitor
from parking_detector import ParkingAreaMonitor



class SimpleMonitor:
    """Base class for simplified monitors that don't require database"""
    
    def __init__(self, system):
        self.system = system
    
    def process_frame(self, frame, objects, zones):
        """Default implementation just returns an empty list"""
        return []
    
    def annotate_frame(self, frame, events):
        """Default implementation returns the frame unmodified"""
        return frame

class EntryAreaMonitor(SimpleMonitor):
    """Simplified entry area monitor"""
    
    def process_frame(self, frame, objects, zones):
        # Detect people sitting
        events = []
        for obj in objects:
            if obj['type'] == 'person' and obj.get('pose') == 'sitting':
                events.append({
                    'event_type': 'sitting_detection',
                    'confidence': obj['confidence_score'],
                    'bounding_box': obj['bounding_box']
                })
        return events
    
    def annotate_frame(self, frame, events):
        # Draw annotations for entry area events
        annotated = frame.copy()
        for event in events:
            if event['event_type'] == 'sitting_detection':
                bbox = event['bounding_box']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated, "Person Sitting", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return annotated

class ReceptionAreaMonitor(SimpleMonitor):
    """Simplified reception area monitor"""
    
    def process_frame(self, frame, objects, zones):
        # Count people in reception
        person_count = sum(1 for obj in objects if obj['type'] == 'person')
        events = []
        if person_count > 5:  # Simplified threshold
            events.append({
                'event_type': 'crowding_detection',
                'person_count': person_count
            })
        return events
    
    def annotate_frame(self, frame, events):
        # Draw annotations for reception area events
        annotated = frame.copy()
        for event in events:
            if event['event_type'] == 'crowding_detection':
                cv2.putText(annotated, f"Crowding: {event['person_count']} people", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return annotated

# Create simplified versions of other monitors
class DataCenterEntryMonitor(SimpleMonitor):
    """Simplified datacenter entry monitor"""
    pass

class DataCenterInsideMonitor(SimpleMonitor):
    """Simplified datacenter inside monitor"""
    pass

class ParkingAreaMonitor(SimpleMonitor):
    """Simplified parking area monitor"""
    pass

class DataCenterSecurityApplication:
    """Main application for data center security monitoring (local version)"""
    
    def __init__(self, config_path: str):
        """
        Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize core system
        self.core_system = DataCenterMonitoringSystem(config_path)
        
        # Initialize simplified monitors
        self.entry_monitor = EntryAreaMonitor(self.core_system)
        self.reception_monitor = ReceptionAreaMonitor(self.core_system)
        self.dc_entry_monitor = DataCenterEntryMonitor(self.core_system)
        self.dc_inside_monitor = DataCenterInsideMonitor(self.core_system)
        self.parking_monitor = ParkingAreaMonitor(self.core_system)
        
        # Get zones configuration
        with open(self.config['zones_config'], 'r') as f:
            self.zones = json.load(f)
    
    def process_video(self, video_path: str, camera_type: str, output_path: str = None):
        """
        Process a video file.
        
        Args:
            video_path: Path to input video file
            camera_type: Type of camera ('entry', 'reception', 'datacenter_entry', 
                                      'datacenter_inside', 'parking')
            output_path: Optional path for output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup output writer if needed
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Get the appropriate zones based on camera type
        if camera_type not in self.zones:
            print(f"Error: Camera type '{camera_type}' not found in zones configuration. Available types: {list(self.zones.keys())}")
            return
            
        camera_zones = self.zones[camera_type]['zones']
        camera_id = camera_type  # Use camera type as ID
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process every nth frame for efficiency
            if frame_count % self.config.get('frame_skip', 1) == 0:
                try:
                    # Detect and track objects using core system
                    annotated_frame, objects = self.core_system.process_frame(frame, camera_id)
                    
                    # Apply specialized monitoring based on camera type
                    specialized_events = []
                    
                    if camera_type == 'entry':
                        events = self.entry_monitor.process_frame(frame, objects, camera_zones)
                        annotated_frame = self.entry_monitor.annotate_frame(annotated_frame, events)
                        specialized_events.extend(events)
                        
                    elif camera_type == 'reception':
                        events = self.reception_monitor.process_frame(frame, objects, camera_zones)
                        annotated_frame = self.reception_monitor.annotate_frame(annotated_frame, events)
                        specialized_events.extend(events)
                        
                    elif camera_type == 'datacenter_entry':
                        events = self.dc_entry_monitor.process_frame(frame, objects, camera_zones)
                        annotated_frame = self.dc_entry_monitor.annotate_frame(annotated_frame, events)
                        specialized_events.extend(events)
                        
                    elif camera_type == 'datacenter_inside':
                        events = self.dc_inside_monitor.process_frame(frame, objects, camera_zones)
                        annotated_frame = self.dc_inside_monitor.annotate_frame(annotated_frame, events)
                        specialized_events.extend(events)
                        
                    elif camera_type == 'parking':
                        events = self.parking_monitor.process_frame(frame, objects, camera_zones)
                        annotated_frame = self.parking_monitor.annotate_frame(annotated_frame, events)
                        specialized_events.extend(events)
                    
                    # Write to output video
                    if writer:
                        writer.write(annotated_frame)
                    
                    # Display frame (always display for local testing)
                    cv2.imshow(f'Monitoring: {camera_type}', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                            
                    # Print events for debugging
                    if specialized_events:
                        for event in specialized_events:
                            print(f"[{camera_type}] Event detected: {event.get('event_type')}")
                            
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Show processing rate every 10 frames
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} fps)")
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
            
        cv2.destroyAllWindows()
            
        print(f"Finished processing {frame_count} frames")
    
    def run_live_camera(self, camera_url: str, camera_type: str):
        """
        Process live camera feed.
        
        Args:
            camera_url: URL or ID of the camera
            camera_type: Type of camera ('entry', 'reception', 'datacenter_entry', 
                                      'datacenter_inside', 'parking')
        """
        # Similar to process_video but for live feed
        cap = cv2.VideoCapture(camera_url)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_url}")
            return
            
        # Get the appropriate zones based on camera type
        if camera_type not in self.zones:
            print(f"Error: Camera type '{camera_type}' not found in zones configuration")
            return
            
        camera_zones = self.zones[camera_type]['zones']
        camera_id = camera_type  # Use camera type as ID
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                    
                frame_count += 1
                
                # Process every nth frame for efficiency
                if frame_count % self.config.get('frame_skip', 1) == 0:
                    try:
                        # Detect and track objects using core system
                        annotated_frame, objects = self.core_system.process_frame(frame, camera_id)
                        
                        # Apply specialized monitoring based on camera type
                        specialized_events = []
                        
                        if camera_type == 'entry':
                            events = self.entry_monitor.process_frame(frame, objects, camera_zones)
                            annotated_frame = self.entry_monitor.annotate_frame(annotated_frame, events)
                            specialized_events.extend(events)
                            
                        elif camera_type == 'reception':
                            events = self.reception_monitor.process_frame(frame, objects, camera_zones)
                            annotated_frame = self.reception_monitor.annotate_frame(annotated_frame, events)
                            specialized_events.extend(events)
                            
                        elif camera_type == 'datacenter_entry':
                            events = self.dc_entry_monitor.process_frame(frame, objects, camera_zones)
                            annotated_frame = self.dc_entry_monitor.annotate_frame(annotated_frame, events)
                            specialized_events.extend(events)
                            
                        elif camera_type == 'datacenter_inside':
                            events = self.dc_inside_monitor.process_frame(frame, objects, camera_zones)
                            annotated_frame = self.dc_inside_monitor.annotate_frame(annotated_frame, events)
                            specialized_events.extend(events)
                            
                        elif camera_type == 'parking':
                            events = self.parking_monitor.process_frame(frame, objects, camera_zones)
                            annotated_frame = self.parking_monitor.annotate_frame(annotated_frame, events)
                            specialized_events.extend(events)
                        
                        # Display frame
                        cv2.imshow(f'Live Monitoring: {camera_type}', annotated_frame)
                        
                        # Check for quit command
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                        # Print events for debugging
                        if specialized_events:
                            for event in specialized_events:
                                print(f"[{camera_type}] Event detected: {event.get('event_type')}")
                                
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")
                
                # Show processing rate every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {frame_count} frames in {elapsed:.2f} seconds ({frame_count/elapsed:.2f} fps)")
        
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"Finished processing {frame_count} frames")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Data Center Security Monitoring (Local Version)')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['video', 'live'], required=True,
                      help='Processing mode: video or live camera')
    parser.add_argument('--input', type=str, required=True, 
                      help='Path to video file or camera URL/ID')
    parser.add_argument('--camera_type', type=str, required=True,
                      choices=['entry', 'reception', 'datacenter_entry', 
                              'datacenter_inside', 'parking'],
                      help='Type of camera being processed')
    parser.add_argument('--output', type=str, help='Path to output video (video mode only)')
    
    args = parser.parse_args()
    
    # Initialize application
    app = DataCenterSecurityApplication(args.config)
    
    # Run in appropriate mode
    if args.mode == 'video':
        app.process_video(args.input, args.camera_type, args.output)
    else:  # Live mode
        app.run_live_camera(args.input, args.camera_type)


if __name__ == "__main__":
    main()


'''
1. For Entry usecase : python3 main.py --config config.json --mode video --input "people_gathering.mp4" --camera_type entry --output output_entry.mp4
2. For Reception usecase : python3 main.py --config config.json --mode video --input "people_gathering.mp4" --camera_type reception --output output_reception.mp4
3. For Data Center Entry usecase : python3 main.py --config config.json --mode video --input "people_gathering.mp4" --camera_type datacenter_entry --output output_dc_entry.mp4
4. For Data Center Inside use case: python3 main.py --config config.json --mode video --input "people_gathering.mp4" --camera_type datacenter_inside --output output_dc_inside.mp4
5. For Parking usecase: python3 main.py --config config.json --mode video --input "people_gathering.mp4" --camera_type parking --output output_parking.mp4

'''