# -*- coding: utf-8 -*-
"""
Video-based Dwell Time Analysis using Shapelet Method
This script processes video files or live camera feeds to extract dwell time series and performs shapelet-based analysis.

Usage:
    # GUI mode (default):
    python liveshapelet_method.py
    
    # CLI mode with video file:
    python liveshapelet_method.py video.mp4
    
    # CLI mode with live camera:
    python liveshapelet_method.py --camera [camera_index]
    
    # CLI mode with camera index 0:
    python liveshapelet_method.py --camera 0
"""

import numpy as np
import pandas as pd
import cv2
import os
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam

from tslearn.shapelets import LearningShapelets
from tslearn.preprocessing import TimeSeriesScalerMinMax

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import random
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_theme()

# Try to import tkinter for file dialog, fallback to command line if not available
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import tkinter.ttk as ttk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Note: tkinter not available. Please provide video path as command line argument.")

# Try to import YOLO for better person detection
try:
    from ultralytics import YOLO
    HAS_YOLO = True
    YOLO_MODEL = None  # Will be loaded on first use
except ImportError:
    HAS_YOLO = False
    print("Note: YOLO (ultralytics) not available. Install with: pip install ultralytics")
    print("Falling back to HOG detection.")

"""# 1. Visitor/Person Tracking Functions"""

def apply_nms(detections, scores=None, iou_threshold=0.7):
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    Keeps the detection with highest score/confidence when boxes overlap.
    
    Parameters:
    -----------
    detections : list
        List of bounding boxes [(x, y, w, h), ...]
    scores : list, optional
        Confidence scores for each detection. If None, uses area as proxy.
    iou_threshold : float
        IoU threshold above which boxes are considered overlapping (default: 0.7)
    
    Returns:
    --------
    filtered_detections : list
        List of non-overlapping bounding boxes
    """
    if len(detections) == 0:
        return []
    
    # Convert to numpy for easier manipulation
    boxes = np.array(detections)
    
    # Calculate scores (use area if scores not provided)
    if scores is None:
        scores = boxes[:, 2] * boxes[:, 3]  # width * height as proxy
    
    scores = np.array(scores)
    
    # Sort by score (highest first)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        
        keep.append(i)
        x1_i, y1_i, w_i, h_i = boxes[i]
        x2_i = x1_i + w_i
        y2_i = y1_i + h_i
        
        # Check IoU with all remaining boxes
        for j in indices:
            if j == i or j in suppressed:
                continue
            
            x1_j, y1_j, w_j, h_j = boxes[j]
            x2_j = x1_j + w_j
            y2_j = y1_j + h_j
            
            # Calculate IoU
            inter_x_min = max(x1_i, x1_j)
            inter_y_min = max(y1_i, y1_j)
            inter_x_max = min(x2_i, x2_j)
            inter_y_max = min(y2_i, y2_j)
            
            if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                box_i_area = w_i * h_i
                box_j_area = w_j * h_j
                union_area = box_i_area + box_j_area - inter_area
                
                if union_area > 0:
                    iou = inter_area / union_area
                    
                    if iou > iou_threshold:
                        suppressed.add(j)
    
    # Return filtered detections
    return [detections[i] for i in keep]

class PersonTracker:
    """Track individual people/visitors in video and calculate their dwell times.
    Only counts frames when human is visible in frame. When human leaves, saves time and resets."""
    
    def __init__(self):
        self.trackers = {}  # Dictionary to store trackers: {person_id: tracker}
        self.person_data = {}  # Dictionary to store person data
        # Structure: {person_id: {
        #   'current_frames': [],  # Frames in current visibility session
        #   'total_frames': 0,      # Total accumulated frames across all sessions
        #   'bbox': None,
        #   'first_seen': None,
        #   'last_seen': None,
        #   'is_visible': False     # Whether person is currently visible
        # }}
        self.next_id = 0
        self.fps = 30  # Will be updated from video
        self.frame_count = 0
        self.completed_sessions = []  # Store completed visit sessions
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update_trackers(self, frame, detections):
        """Update existing trackers and create new ones for new detections.
        Only counts frames when human is visible. Resets count when human leaves.
        Enhanced duplicate prevention using IoU matching and distance-based tracking."""
        # Use IoU-based tracking with enhanced duplicate prevention
        active_ids = list(self.person_data.keys())
        matched_detections = set()
        
        # Calculate center points for distance-based matching (additional duplicate prevention)
        def get_center(bbox):
            if isinstance(bbox, tuple) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                return ((x1 + x2) / 2, (y1 + y2) / 2)
            return None
        
        # Check for people who left the frame
        for person_id in active_ids:
            if person_id not in self.person_data:
                continue
            
            # If person was visible but not detected in this frame
            if self.person_data[person_id]['is_visible']:
                # Check if they're still in detections
                existing_bbox = self.person_data[person_id]['bbox']
                if existing_bbox is None:
                    self.person_data[person_id]['is_visible'] = False
                    continue
                
                # Find best matching detection using IoU and distance
                best_match_idx = None
                best_iou = 0.4  # Minimum IoU threshold for matching
                best_score = 0
                
                existing_center = get_center(existing_bbox)
                
                for idx, det_bbox in enumerate(detections):
                    if idx in matched_detections:
                        continue
                        
                    x, y, w, h = det_bbox
                    det_box = (x, y, x+w, y+h)
                    iou = self.calculate_iou(det_box, existing_bbox)
                    
                    # Calculate distance between centers (normalized)
                    det_center = get_center(det_box)
                    if existing_center and det_center:
                        # Normalize distance by frame dimensions (approximate)
                        dist = np.sqrt((existing_center[0] - det_center[0])**2 + 
                                     (existing_center[1] - det_center[1])**2)
                        # Combine IoU and distance (prefer higher IoU, lower distance)
                        score = iou * 0.7 + (1.0 / (1.0 + dist / 100.0)) * 0.3
                    else:
                        score = iou
                    
                    if iou > best_iou and score > best_score:
                        best_iou = iou
                        best_score = score
                        best_match_idx = idx
                
                if best_match_idx is None:
                    # Person not detected in this frame
                    frames_since_seen = self.frame_count - self.person_data[person_id]['last_seen']
                    if frames_since_seen > 15:  # Threshold for person leaving frame
                        # Save current session time and reset
                        current_frames = len(self.person_data[person_id]['current_frames'])
                        if current_frames > 0:
                            # Add to total frames
                            self.person_data[person_id]['total_frames'] += current_frames
                            
                            # Save completed session
                            self.completed_sessions.append({
                                'person_id': person_id,
                                'frames': current_frames,
                                'start_frame': self.person_data[person_id]['current_frames'][0] if self.person_data[person_id]['current_frames'] else self.person_data[person_id]['first_seen'],
                                'end_frame': self.person_data[person_id]['last_seen']
                            })
                            
                            # Reset current session
                            self.person_data[person_id]['current_frames'] = []
                            self.person_data[person_id]['is_visible'] = False
                            self.person_data[person_id]['bbox'] = None
        
        # Try to match detections to existing people (enhanced matching)
        # Sort by person ID to ensure consistent matching
        for person_id in sorted(active_ids):
            if person_id not in self.person_data:
                continue
                
            existing_bbox = self.person_data[person_id]['bbox']
            
            # Find best matching detection using IoU and distance
            best_match_idx = None
            best_iou = 0.5  # Minimum IoU threshold for matching existing person
            best_score = 0
            
            existing_center = get_center(existing_bbox)
            
            for idx, det_bbox in enumerate(detections):
                if idx in matched_detections:
                    continue
                    
                x, y, w, h = det_bbox
                det_box = (x, y, x+w, y+h)
                
                if existing_bbox:
                    iou = self.calculate_iou(det_box, existing_bbox)
                    
                    # Calculate distance between centers
                    det_center = get_center(det_box)
                    if existing_center and det_center:
                        dist = np.sqrt((existing_center[0] - det_center[0])**2 + 
                                     (existing_center[1] - det_center[1])**2)
                        # Combined score: IoU weighted more heavily
                        score = iou * 0.7 + (1.0 / (1.0 + dist / 100.0)) * 0.3
                    else:
                        score = iou
                else:
                    # No existing bbox, skip this person
                    continue
                
                if iou > best_iou and score > best_score:
                    best_iou = iou
                    best_score = score
                    best_match_idx = idx
            
            if best_match_idx is not None:
                # Update existing person (visible in this frame) - NO DUPLICATE
                x, y, w, h = detections[best_match_idx]
                det_box = (x, y, x+w, y+h)
                
                # If person was not visible, start new session
                if not self.person_data[person_id]['is_visible']:
                    self.person_data[person_id]['first_seen'] = self.frame_count
                
                self.person_data[person_id]['bbox'] = det_box
                self.person_data[person_id]['last_seen'] = self.frame_count
                self.person_data[person_id]['current_frames'].append(self.frame_count)  # Only count when visible
                self.person_data[person_id]['is_visible'] = True
                matched_detections.add(best_match_idx)
        
        # Create new trackers ONLY for unmatched detections (new people entering)
        # This ensures no duplicates - each detection is matched to at most one person
        for idx, det_bbox in enumerate(detections):
            if idx not in matched_detections:
                x, y, w, h = det_bbox
                det_box = (x, y, x+w, y+h)
                
                # Check if this detection might be a duplicate of an inactive person
                # (person who left but came back - should reuse ID)
                is_duplicate = False
                for person_id in active_ids:
                    if person_id not in self.person_data:
                        continue
                    if not self.person_data[person_id]['is_visible']:
                        existing_bbox = self.person_data[person_id]['bbox']
                        if existing_bbox:
                            iou = self.calculate_iou(det_box, existing_bbox)
                            if iou > 0.3:  # Might be same person returning
                                # Reuse existing ID
                                self.person_data[person_id]['bbox'] = det_box
                                self.person_data[person_id]['last_seen'] = self.frame_count
                                self.person_data[person_id]['first_seen'] = self.frame_count
                                self.person_data[person_id]['current_frames'] = [self.frame_count]
                                self.person_data[person_id]['is_visible'] = True
                                is_duplicate = True
                                matched_detections.add(idx)
                                break
                
                if not is_duplicate:
                    # New person - assign new ID
                    person_id = self.next_id
                    self.next_id += 1
                    self.trackers[person_id] = True  # Placeholder
                    self.person_data[person_id] = {
                        'current_frames': [self.frame_count],  # Start counting frames
                        'total_frames': 0,  # Will accumulate across sessions
                        'bbox': det_box,
                        'first_seen': self.frame_count,
                        'last_seen': self.frame_count,
                        'is_visible': True  # Currently visible
                    }
    
    def get_visitor_times(self):
        """Calculate time spent by each visitor.
        Only counts frames when human is visible. Includes current session + completed sessions."""
        visitor_times = []
        
        # Process all people (active and completed)
        all_person_ids = set(self.person_data.keys())
        
        # Add completed sessions
        for session in self.completed_sessions:
            all_person_ids.add(session['person_id'])
        
        for person_id in all_person_ids:
            # Get current frames (if person is still visible)
            current_frames = 0
            if person_id in self.person_data:
                current_frames = len(self.person_data[person_id]['current_frames'])
                total_frames = self.person_data[person_id]['total_frames'] + current_frames
                first_seen = self.person_data[person_id]['first_seen']
                last_seen = self.person_data[person_id]['last_seen']
            else:
                # Person only exists in completed sessions
                total_frames = 0
                first_seen = None
                last_seen = None
            
            # Add frames from completed sessions for this person
            for session in self.completed_sessions:
                if session['person_id'] == person_id:
                    total_frames += session['frames']
                    if first_seen is None or session['start_frame'] < first_seen:
                        first_seen = session['start_frame']
                    if last_seen is None or session['end_frame'] > last_seen:
                        last_seen = session['end_frame']
            
            if total_frames > 0:
                time_seconds = total_frames / self.fps if self.fps > 0 else 0
                time_minutes = time_seconds / 60
                
                visitor_times.append({
                    'person_id': person_id,
                    'frames': total_frames,
                    'current_frames': current_frames,  # Frames in current session
                    'time_seconds': time_seconds,
                    'time_minutes': time_minutes,
                    'time_formatted': f"{int(time_minutes)}m {int(time_seconds % 60)}s",
                    'first_seen_frame': first_seen,
                    'last_seen_frame': last_seen,
                    'is_currently_visible': person_id in self.person_data and self.person_data[person_id]['is_visible']
                })
        
        return sorted(visitor_times, key=lambda x: x['time_seconds'], reverse=True)

def detect_humans_yolo(frame, model=None, confidence=0.25):
    """
    Detect humans using YOLO model (best accuracy).
    YOLO is state-of-the-art for object detection including people.
    Applies NMS to remove overlapping detections.
    
    Parameters:
    -----------
    frame : numpy array
        Video frame
    model : YOLO model, optional
        Pre-loaded YOLO model. If None, will load on first call.
    confidence : float
        Confidence threshold (0.0 to 1.0)
    
    Returns:
    --------
    detections : list
        List of bounding boxes [(x, y, w, h), ...] after NMS
    """
    global YOLO_MODEL
    
    if not HAS_YOLO:
        return []
    
    detections = []
    scores = []
    
    try:
        # Load model on first use
        if model is None:
            if YOLO_MODEL is None:
                print("Loading YOLO model for person detection...")
                YOLO_MODEL = YOLO('yolov8n.pt')  # nano model - fast and accurate
                print("YOLO model loaded successfully!")
            model = YOLO_MODEL
        
        # Run detection with NMS enabled in YOLO (but we'll apply additional NMS)
        results = model(frame, conf=confidence, classes=[0], verbose=False, iou=0.7)  # class 0 = person
        
        # Extract bounding boxes and confidence scores
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence score
                conf_score = float(box.conf[0].cpu().numpy())
                
                # Convert to (x, y, w, h) format
                w = x2 - x1
                h = y2 - y1
                
                # Filter by size (remove very small detections)
                min_size = 30  # Minimum width or height
                if w >= min_size and h >= min_size:
                    detections.append((x1, y1, w, h))
                    scores.append(conf_score)
        
        # Apply NMS to remove overlapping detections
        if len(detections) > 1:
            detections = apply_nms(detections, scores=scores, iou_threshold=0.7)
        
    except Exception as e:
        print(f"Warning: YOLO detection failed: {e}")
        return []
    
    return detections

def detect_humans_hog(frame):
    """
    Detect humans using HOG (Histogram of Oriented Gradients) descriptor.
    This is specifically designed for human detection.
    
    Parameters:
    -----------
    frame : numpy array
        Video frame
    
    Returns:
    --------
    detections : list
        List of bounding boxes [(x, y, w, h), ...]
    """
    detections = []
    
    try:
        # Initialize HOG descriptor for person detection
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Detect humans in the frame
        # winStride: step size for sliding window
        # padding: padding around the image
        # scale: scale factor for multi-scale detection
        (rects, weights) = hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=0.0,  # Lower threshold for better detection
            finalThreshold=2.0  # Final threshold after non-maximum suppression
        )
        
        # Filter detections based on human body proportions
        for (x, y, w, h) in rects:
            # Human-specific filtering
            aspect_ratio = h / w if w > 0 else 0
            
            # Typical human aspect ratio: 1.5 to 3.5 (taller than wide)
            # Typical human size: height should be reasonable relative to frame
            if 1.2 < aspect_ratio < 4.0:  # Human body proportions
                # Additional validation: check if detection makes sense
                # Minimum reasonable size for a person
                min_height = frame.shape[0] * 0.1  # At least 10% of frame height
                max_height = frame.shape[0] * 0.9  # At most 90% of frame height
                
                if min_height < h < max_height:
                    detections.append((x, y, w, h))
        
    except Exception as e:
        print(f"Warning: HOG detection failed: {e}")
        # Fall back to contour-based detection
        return detect_humans_contours(frame)
    
    return detections

def detect_humans_contours(frame, fg_mask=None):
    """
    Detect humans using contour analysis with human-specific filtering.
    
    Parameters:
    -----------
    frame : numpy array
        Video frame
    fg_mask : numpy array, optional
        Foreground mask from background subtraction
    
    Returns:
    --------
    detections : list
        List of bounding boxes [(x, y, w, h), ...]
    """
    detections = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if fg_mask is None:
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Use adaptive threshold
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        thresh = fg_mask
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Human-specific filtering
    frame_height, frame_width = frame.shape[:2]
    
    # Typical human proportions and sizes
    min_area = frame_height * frame_width * 0.01  # At least 1% of frame
    max_area = frame_height * frame_width * 0.3   # At most 30% of frame
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Human body proportion checks
            aspect_ratio = h / w if w > 0 else 0
            
            # Humans are typically taller than wide
            # Typical range: 1.5 to 3.5 for standing person
            # Allow wider range for different poses: 1.0 to 4.0
            if 1.0 < aspect_ratio < 4.0:
                # Additional checks for human-like dimensions
                # Height should be reasonable
                min_height = frame_height * 0.08  # At least 8% of frame height
                max_height = frame_height * 0.85  # At most 85% of frame height
                
                # Width should be reasonable (not too wide for a person)
                max_width = frame_width * 0.4  # Person shouldn't be wider than 40% of frame
                
                if min_height < h < max_height and w < max_width:
                    # Check solidity (how filled the contour is)
                    # Humans have relatively solid shapes
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Humans typically have solidity > 0.5
                    if solidity > 0.4:
                        detections.append((x, y, w, h))
    
    return detections

def detect_people_in_frame(frame, method='yolo', fg_mask=None, yolo_model=None):
    """
    Detect people/visitors in a single frame with human-specific filtering.
    Uses YOLO by default for best accuracy, falls back to HOG if YOLO unavailable.
    
    Parameters:
    -----------
    frame : numpy array
        Video frame
    method : str
        Detection method ('yolo', 'hog', 'contours', 'combined')
    fg_mask : numpy array, optional
        Foreground mask from background subtraction
    yolo_model : YOLO model, optional
        Pre-loaded YOLO model
    
    Returns:
    --------
    detections : list
        List of bounding boxes [(x, y, w, h), ...]
    """
    # Try YOLO first (best accuracy)
    if method == 'yolo' or (method == 'auto' and HAS_YOLO):
        yolo_detections = detect_humans_yolo(frame, model=yolo_model, confidence=0.25)
        if len(yolo_detections) > 0:
            return yolo_detections
        # If YOLO found nothing but is available, still return empty (might be no people)
        if HAS_YOLO:
            return yolo_detections
    
    # Fallback methods
    if method == 'hog' or (method == 'auto' and not HAS_YOLO):
        return detect_humans_hog(frame)
    elif method == 'contours':
        return detect_humans_contours(frame, fg_mask)
    elif method == 'combined':
        # Use multiple methods and combine results with proper NMS
        all_detections = []
        all_scores = []
        
        # Try YOLO first (most accurate)
        if HAS_YOLO:
            yolo_detections = detect_humans_yolo(frame, model=yolo_model, confidence=0.2)
            all_detections.extend(yolo_detections)
            # YOLO detections already have high confidence, assign score 0.8-1.0
            all_scores.extend([0.9] * len(yolo_detections))
        
        # Add HOG detections (medium confidence)
        hog_detections = detect_humans_hog(frame)
        all_detections.extend(hog_detections)
        all_scores.extend([0.6] * len(hog_detections))
        
        # Add contour detections (lower confidence)
        contour_detections = detect_humans_contours(frame, fg_mask)
        all_detections.extend(contour_detections)
        all_scores.extend([0.4] * len(contour_detections))
        
        # Apply NMS with higher threshold (0.7) to remove duplicates
        if len(all_detections) > 1:
            filtered_detections = apply_nms(all_detections, scores=all_scores, iou_threshold=0.7)
        else:
            filtered_detections = all_detections
        
        return filtered_detections
    else:
        # Default: try YOLO, fallback to HOG
        if HAS_YOLO:
            return detect_humans_yolo(frame, model=yolo_model, confidence=0.25)
        else:
            return detect_humans_hog(frame)

def draw_info_panel(frame, tracker, frame_count, total_frames, fps):
    """Draw information panel on the frame showing live statistics."""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for info panel
    overlay = frame.copy()
    panel_height = 200
    cv2.rectangle(overlay, (10, 10), (width - 10, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Get current visitor statistics
    visitor_times = tracker.get_visitor_times()
    # Count only currently visible visitors
    active_visitors = len([v for v in visitor_times if v.get('is_currently_visible', False)])
    
    # Title
    cv2.putText(frame, "HUMAN DWELL TIME ANALYSIS - LIVE PROCESSING", 
               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Statistics
    y_offset = 60
    line_height = 25
    
    # Progress
    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
    cv2.putText(frame, f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += line_height
    
    # Current time
    current_time_sec = frame_count / fps if fps > 0 else 0
    current_time_min = int(current_time_sec // 60)
    current_time_sec_remainder = int(current_time_sec % 60)
    cv2.putText(frame, f"Video Time: {current_time_min:02d}:{current_time_sec_remainder:02d}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += line_height
    
    # Total visitors
    cv2.putText(frame, f"Total Visitors Detected: {tracker.next_id}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    y_offset += line_height
    
    # Active visitors
    cv2.putText(frame, f"Active Visitors (current frame): {active_visitors}", 
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    y_offset += line_height
    
    # Total time
    if len(visitor_times) > 0:
        total_time = sum(v['time_seconds'] for v in visitor_times)
        avg_time = np.mean([v['time_seconds'] for v in visitor_times])
        cv2.putText(frame, f"Total Time (all visitors): {total_time/60:.1f} min | Avg: {avg_time:.1f}s", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    return frame

def track_visitors_in_camera(camera_index=0, output_video_path=None, show_preview=True):
    """
    Track visitors/people in live camera feed and calculate time spent by each.
    
    Parameters:
    -----------
    camera_index : int
        Camera device index (default: 0)
    output_video_path : str, optional
        Path to save annotated video
    show_preview : bool
        Whether to show preview during processing
    
    Returns:
    --------
    visitor_times : list
        List of dictionaries with visitor time data
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening camera device: {camera_index}")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS if camera doesn't report it
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n{'='*60}")
    print(f"LIVE CAMERA FEED")
    print(f"{'='*60}")
    print(f"  Camera Index: {camera_index}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Press 'q' to stop, 'p' to pause/resume")
    print(f"{'='*60}\n")
    
    # Initialize tracker
    tracker = PersonTracker()
    tracker.fps = fps
    
    # Initialize YOLO model if available
    # Try to load trained premises model first, fallback to default
    yolo_model = None
    detection_method = 'auto'  # Will try YOLO first, fallback to HOG
    
    if HAS_YOLO:
        try:
            # Try to load trained premises-specific model first
            trained_model_paths = [
                'trained_models/ranka_premises_best.pt',
                'trained_models/yolo_dataset/ranka_premises/weights/best.pt',
                'ranka_premises_best.pt'
            ]
            
            model_loaded = False
            for model_path in trained_model_paths:
                if os.path.exists(model_path):
                    print(f"Loading trained premises model: {model_path}")
                    yolo_model = YOLO(model_path)
                    detection_method = 'yolo'
                    print("✓ Trained premises model loaded - optimized for this location")
                    model_loaded = True
                    break
            
            if not model_loaded:
                print("Initializing YOLO model for accurate person detection...")
                yolo_model = YOLO('yolov8n.pt')  # nano model - good balance of speed and accuracy
                detection_method = 'yolo'
                print("✓ YOLO model loaded - using advanced person detection")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            print("Falling back to HOG detection")
            detection_method = 'hog'
            yolo_model = None
    else:
        print("YOLO not available - using HOG detection")
        print("For better accuracy, install YOLO: pip install ultralytics")
        detection_method = 'hog'
    
    # Initialize background subtractor for better detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        detectShadows=True, 
        history=500, 
        varThreshold=40
    )
    
    # Video writer for output
    out = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path) if os.path.dirname(output_video_path) else '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Saving annotated video to: {output_video_path}\n")
    
    frame_count = 0
    
    print("Starting live camera processing with HUMAN DETECTION...")
    if detection_method == 'yolo' and yolo_model is not None:
        print("Using YOLO (You Only Look Once) - State-of-the-art person detection")
        print("✓ YOLO model loaded - Maximum accuracy for detecting all people")
    else:
        print("Using HOG (Histogram of Oriented Gradients) for human detection")
        print("Note: For better accuracy, YOLO is recommended")
    print("Press 'q' to quit, 'p' to pause/resume\n")
    
    paused = False
    
    # Color palette for different visitors
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera")
                break
            
            tracker.frame_count = frame_count
            
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Detect humans using YOLO (best accuracy) or fallback methods
            try:
                if detection_method == 'yolo' and yolo_model is not None:
                    detections = detect_people_in_frame(frame, method='yolo', fg_mask=None, yolo_model=yolo_model)
                    if len(detections) < 2:
                        combined_detections = detect_people_in_frame(frame, method='combined', fg_mask=fg_mask, yolo_model=yolo_model)
                        all_detections = detections + combined_detections
                        if len(all_detections) > 1:
                            detections = apply_nms(all_detections, iou_threshold=0.7)
                        else:
                            detections = all_detections
                else:
                    detections = detect_people_in_frame(frame, method='combined', fg_mask=fg_mask, yolo_model=yolo_model)
            except Exception as e:
                print(f"Warning: Detection error: {e}")
                detections = detect_people_in_frame(frame, method='contours', fg_mask=fg_mask)
            
            # Additional filtering
            filtered_detections = []
            min_human_height = height * 0.06
            min_human_width = width * 0.04
            min_area_ratio = 0.003
            
            for det in detections:
                x, y, w, h = det
                if h >= min_human_height and w >= min_human_width:
                    if h * w >= (height * width * min_area_ratio):
                        aspect_ratio = h / w if w > 0 else 0
                        if 0.8 < aspect_ratio < 5.0:
                            filtered_detections.append(det)
            
            # Apply final NMS
            if len(filtered_detections) > 1:
                filtered_detections = apply_nms(filtered_detections, iou_threshold=0.7)
            
            detections = filtered_detections
            
            # Update trackers
            if len(detections) > 0:
                tracker.update_trackers(frame, detections)
            
            # Draw bounding boxes and IDs on frame
            annotated_frame = frame.copy()
            
            # Draw each tracked person
            for person_id, data in tracker.person_data.items():
                if data['bbox']:
                    x1, y1, x2, y2 = data['bbox']
                    
                    # Choose color for this person
                    color = colors[person_id % len(colors)]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Calculate time spent
                    current_frames = len(data['current_frames'])
                    total_frames = data['total_frames'] + current_frames
                    time_sec = total_frames / fps if fps > 0 else 0
                    time_min = int(time_sec // 60)
                    time_sec_remainder = int(time_sec % 60)
                    
                    # Draw background for text
                    text_bg_height = 60
                    cv2.rectangle(annotated_frame, (x1, y1 - text_bg_height), 
                                 (x1 + 200, y1), color, -1)
                    
                    # Draw person ID
                    cv2.putText(annotated_frame, f"Visitor #{person_id}", 
                               (x1 + 5, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    
                    # Draw time spent
                    time_text = f"Time: {time_min:02d}:{time_sec_remainder:02d}"
                    cv2.putText(annotated_frame, time_text, 
                               (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                    
                    # Draw frame count
                    if data['total_frames'] > 0:
                        frame_text = f"Frames: {current_frames} (Total: {total_frames})"
                    else:
                        frame_text = f"Frames: {current_frames}"
                    cv2.putText(annotated_frame, frame_text, 
                               (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, color, 2)
            
            # Draw information panel (for live camera, use frame_count as total_frames)
            annotated_frame = draw_info_panel(annotated_frame, tracker, frame_count, frame_count, fps)
            
            # Write frame if output video specified
            if out:
                out.write(annotated_frame)
            
            frame_count += 1
        
        # Show preview
        if show_preview:
            cv2.imshow('Live Camera - Visitor Dwell Time Analysis', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nProcessing stopped by user.")
                break
            elif key == ord('p'):
                paused = not paused
                if paused:
                    print("Paused. Press 'p' to resume.")
                else:
                    print("Resumed.")
        
        # Progress indicator in console (every second)
        if frame_count % max(1, int(fps)) == 0:
            visitor_times = tracker.get_visitor_times()
            active_count = len([v for v in visitor_times if v.get('is_currently_visible', False)])
            print(f"\rFrame: {frame_count} | "
                  f"Visitors: {tracker.next_id} | Active: {active_count}", end='', flush=True)
    
    # Save any remaining active sessions before finishing
    for person_id, data in tracker.person_data.items():
        if data['is_visible'] and len(data['current_frames']) > 0:
            current_frames = len(data['current_frames'])
            tracker.person_data[person_id]['total_frames'] += current_frames
            
            tracker.completed_sessions.append({
                'person_id': person_id,
                'frames': current_frames,
                'start_frame': data['current_frames'][0] if data['current_frames'] else data['first_seen'],
                'end_frame': data['last_seen']
            })
            
            tracker.person_data[person_id]['current_frames'] = []
            tracker.person_data[person_id]['is_visible'] = False
    
    cap.release()
    if out:
        out.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"\n\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total visitors tracked: {tracker.next_id}")
    print(f"Total frames processed: {frame_count}")
    print(f"Completed sessions: {len(tracker.completed_sessions)}")
    print(f"{'='*60}\n")
    
    # Get visitor times
    visitor_times = tracker.get_visitor_times()
    
    return visitor_times

def track_visitors_in_video(video_path, output_video_path=None, show_preview=True):
    """
    Track visitors/people in video and calculate time spent by each.
    
    Parameters:
    -----------
    video_path : str
        Path to video file
    output_video_path : str, optional
        Path to save annotated video
    show_preview : bool
        Whether to show preview during processing
    
    Returns:
    --------
    visitor_times : list
        List of dictionaries with visitor time data
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*60}")
    print(f"VIDEO PROPERTIES")
    print(f"{'='*60}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Estimated duration: {total_frames/fps:.1f} seconds")
    print(f"{'='*60}\n")
    
    # Initialize tracker
    tracker = PersonTracker()
    tracker.fps = fps
    
    # Initialize YOLO model if available
    # Try to load trained premises model first, fallback to default
    yolo_model = None
    detection_method = 'auto'  # Will try YOLO first, fallback to HOG
    
    if HAS_YOLO:
        try:
            # Try to load trained premises-specific model first
            trained_model_paths = [
                'trained_models/ranka_premises_best.pt',
                'trained_models/yolo_dataset/ranka_premises/weights/best.pt',
                'ranka_premises_best.pt'
            ]
            
            model_loaded = False
            for model_path in trained_model_paths:
                if os.path.exists(model_path):
                    print(f"Loading trained premises model: {model_path}")
                    yolo_model = YOLO(model_path)
                    detection_method = 'yolo'
                    print("✓ Trained premises model loaded - optimized for this location")
                    model_loaded = True
                    break
            
            if not model_loaded:
                print("Initializing YOLO model for accurate person detection...")
                yolo_model = YOLO('yolov8n.pt')  # nano model - good balance of speed and accuracy
                detection_method = 'yolo'
                print("✓ YOLO model loaded - using advanced person detection")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            print("Falling back to HOG detection")
            detection_method = 'hog'
            yolo_model = None
    else:
        print("YOLO not available - using HOG detection")
        print("For better accuracy, install YOLO: pip install ultralytics")
        detection_method = 'hog'
    
    # Initialize background subtractor for better detection
    # Adjusted parameters for better human detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        detectShadows=True, 
        history=500, 
        varThreshold=40  # Lower threshold for better sensitivity
    )
    
    # Video writer for output
    out = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path) if os.path.dirname(output_video_path) else '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"Saving annotated video to: {output_video_path}\n")
    
    frame_count = 0
    
    print("Starting live processing with HUMAN DETECTION...")
    if detection_method == 'yolo' and yolo_model is not None:
        print("Using YOLO (You Only Look Once) - State-of-the-art person detection")
        print("✓ YOLO model loaded - Maximum accuracy for detecting all people")
    else:
        print("Using HOG (Histogram of Oriented Gradients) for human detection")
        print("Note: For better accuracy, YOLO is recommended")
    print("Press 'q' to quit, 'p' to pause/resume\n")
    
    paused = False
    
    # Color palette for different visitors
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            tracker.frame_count = frame_count
            
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(frame)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Detect humans using YOLO (best accuracy) or fallback methods
            try:
                # Use YOLO if available, otherwise use combined methods for better coverage
                if detection_method == 'yolo' and yolo_model is not None:
                    detections = detect_people_in_frame(frame, method='yolo', fg_mask=None, yolo_model=yolo_model)
                    # If YOLO finds few detections, also try combined method for better coverage
                    if len(detections) < 2:  # If we expect more people
                        combined_detections = detect_people_in_frame(frame, method='combined', fg_mask=fg_mask, yolo_model=yolo_model)
                        # Merge results and apply NMS
                        all_detections = detections + combined_detections
                        if len(all_detections) > 1:
                            detections = apply_nms(all_detections, iou_threshold=0.7)
                        else:
                            detections = all_detections
                else:
                    # Use combined method for maximum coverage
                    detections = detect_people_in_frame(frame, method='combined', fg_mask=fg_mask, yolo_model=yolo_model)
            except Exception as e:
                print(f"Warning: Detection error: {e}")
                # Fall back to contour-based detection
                detections = detect_people_in_frame(frame, method='contours', fg_mask=fg_mask)
            
            # Additional filtering: Remove very small detections (likely noise)
            # and ensure minimum size for human detection
            filtered_detections = []
            min_human_height = height * 0.06  # Increased from 5% to 6% of frame height
            min_human_width = width * 0.04     # Increased from 3% to 4% of frame width
            min_area_ratio = 0.003  # Increased from 0.002 to 0.003 (0.3% of frame area)
            
            for det in detections:
                x, y, w, h = det
                # Filter by minimum size
                if h >= min_human_height and w >= min_human_width:
                    # Additional check: humans shouldn't be too small relative to frame
                    if h * w >= (height * width * min_area_ratio):
                        # Filter unrealistic aspect ratios
                        aspect_ratio = h / w if w > 0 else 0
                        if 0.8 < aspect_ratio < 5.0:  # Reasonable human proportions
                            filtered_detections.append(det)
            
            # Apply final NMS before tracking to ensure no duplicates
            if len(filtered_detections) > 1:
                filtered_detections = apply_nms(filtered_detections, iou_threshold=0.7)
            
            detections = filtered_detections
            
            # Update trackers
            if len(detections) > 0:
                tracker.update_trackers(frame, detections)
            
            # Draw bounding boxes and IDs on frame
            annotated_frame = frame.copy()
            
            # Draw each tracked person
            for person_id, data in tracker.person_data.items():
                if data['bbox']:
                    x1, y1, x2, y2 = data['bbox']
                    
                    # Choose color for this person
                    color = colors[person_id % len(colors)]
                    
                    # Draw bounding box with thicker line
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Calculate time spent (only current visible session)
                    current_frames = len(data['current_frames'])
                    total_frames = data['total_frames'] + current_frames
                    time_sec = total_frames / fps if fps > 0 else 0
                    time_min = int(time_sec // 60)
                    time_sec_remainder = int(time_sec % 60)
                    
                    # Draw background for text (for better visibility)
                    text_bg_height = 60
                    cv2.rectangle(annotated_frame, (x1, y1 - text_bg_height), 
                                 (x1 + 200, y1), color, -1)
                    
                    # Draw person ID
                    cv2.putText(annotated_frame, f"Visitor #{person_id}", 
                               (x1 + 5, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                    
                    # Draw time spent
                    time_text = f"Time: {time_min:02d}:{time_sec_remainder:02d}"
                    cv2.putText(annotated_frame, time_text, 
                               (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (255, 255, 255), 2)
                    
                    # Draw frame count (current session + total)
                    if data['total_frames'] > 0:
                        frame_text = f"Frames: {current_frames} (Total: {total_frames})"
                    else:
                        frame_text = f"Frames: {current_frames}"
                    cv2.putText(annotated_frame, frame_text, 
                               (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, color, 2)
            
            # Draw information panel
            annotated_frame = draw_info_panel(annotated_frame, tracker, frame_count, total_frames, fps)
            
            # Write frame if output video specified
            if out:
                out.write(annotated_frame)
            
            frame_count += 1
        
        # Show preview
        if show_preview:
            cv2.imshow('Visitor Dwell Time Analysis - LIVE PROCESSING', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nProcessing stopped by user.")
                break
            elif key == ord('p'):
                paused = not paused
                if paused:
                    print("Paused. Press 'p' to resume.")
                else:
                    print("Resumed.")
        
        # Progress indicator in console
        if frame_count % max(1, int(fps)) == 0:  # Update every second
            progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
            visitor_times = tracker.get_visitor_times()
            active_count = len([v for v in visitor_times if v.get('is_currently_visible', False)])
            print(f"\rFrame: {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"Visitors: {tracker.next_id} | Active: {active_count}", end='', flush=True)
    
    # Save any remaining active sessions before finishing
    for person_id, data in tracker.person_data.items():
        if data['is_visible'] and len(data['current_frames']) > 0:
            # Save current session
            current_frames = len(data['current_frames'])
            tracker.person_data[person_id]['total_frames'] += current_frames
            
            tracker.completed_sessions.append({
                'person_id': person_id,
                'frames': current_frames,
                'start_frame': data['current_frames'][0] if data['current_frames'] else data['first_seen'],
                'end_frame': data['last_seen']
            })
            
            # Clear current session
            tracker.person_data[person_id]['current_frames'] = []
            tracker.person_data[person_id]['is_visible'] = False
    
    cap.release()
    if out:
        out.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    print(f"\n\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total visitors tracked: {tracker.next_id}")
    print(f"Total frames processed: {frame_count}")
    print(f"Completed sessions: {len(tracker.completed_sessions)}")
    print(f"{'='*60}\n")
    
    # Get visitor times
    visitor_times = tracker.get_visitor_times()
    
    return visitor_times

"""# 2. Video Processing Functions"""

def extract_dwell_times_from_video(video_path, roi=None, threshold_method='adaptive', 
                                   frame_skip=1, min_area=100):
    """
    Extract dwell time series from video by tracking objects/states.
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    roi : tuple, optional
        Region of interest (x, y, width, height). If None, uses full frame
    threshold_method : str
        Method for thresholding ('adaptive', 'otsu', 'simple')
    frame_skip : int
        Process every Nth frame (1 = all frames)
    min_area : int
        Minimum area for detected objects
    
    Returns:
    --------
    dwell_times : numpy array
        Array of dwell times (duration of states)
    frame_data : numpy array
        Processed frame data for visualization
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    dwell_times = []
    frame_data = []
    prev_state = None
    current_state_duration = 0
    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps:.2f}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply ROI if specified
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        
        # Apply thresholding
        if threshold_method == 'adaptive':
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
        elif threshold_method == 'otsu':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate total area of detected objects
        total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > min_area)
        
        # Determine state (normalized area as proxy for state)
        normalized_area = total_area / (gray.shape[0] * gray.shape[1]) if gray.size > 0 else 0
        
        # Classify state (simple binary classification based on threshold)
        current_state = 1 if normalized_area > 0.1 else 0
        
        # Track state changes and accumulate dwell times
        if prev_state is not None:
            if current_state == prev_state:
                current_state_duration += 1
            else:
                if current_state_duration > 0:
                    dwell_times.append(current_state_duration)
                current_state_duration = 1
        else:
            current_state_duration = 1
        
        prev_state = current_state
        frame_data.append(normalized_area)
        frame_count += 1
        
        # Progress indicator
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...", end='\r')
    
    # Add final state duration
    if current_state_duration > 0:
        dwell_times.append(current_state_duration)
    
    cap.release()
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Total dwell time states detected: {len(dwell_times)}")
    
    return np.array(dwell_times), np.array(frame_data)

def extract_dwell_times_simple(video_path, method='intensity'):
    """
    Simplified method: Extract dwell times based on frame intensity changes.
    
    Parameters:
    -----------
    video_path : str
        Path to video file
    method : str
        'intensity' - uses mean intensity per frame
        'motion' - uses optical flow to detect motion
    
    Returns:
    --------
    dwell_times : numpy array
        Dwell time series
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    intensities = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if method == 'intensity':
            mean_intensity = np.mean(gray)
            intensities.append(mean_intensity)
        elif method == 'motion':
            # This would require previous frame - simplified version
            intensities.append(np.std(gray))
    
    cap.release()
    
    # Convert intensities to dwell times by detecting state changes
    intensities = np.array(intensities)
    # Normalize
    intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
    
    # Detect state changes (threshold-based)
    threshold = np.median(intensities)
    states = (intensities > threshold).astype(int)
    
    # Calculate dwell times
    dwell_times = []
    prev_state = states[0]
    duration = 1
    
    for state in states[1:]:
        if state == prev_state:
            duration += 1
        else:
            dwell_times.append(duration)
            duration = 1
        prev_state = state
    
    dwell_times.append(duration)
    
    return np.array(dwell_times), intensities

"""# 3. Time Series Preprocessing Functions"""

def divide(data, length, stride):
    '''
    data: numpy array containing the dwell times series data
    length: length of the subseries after segmentation
    stride: space (in points) between beginnings of two consecutive subseries
    '''
    if len(data) < length:
        # Pad if data is shorter than required length
        data = np.pad(data, (0, length - len(data)), mode='edge')
    
    rslt = []
    i = 0
    while (i * stride < len(data) - length):
        piece = data[i * stride:(i * stride + length)]
        i += 1
        rslt.append(piece)
    
    if len(rslt) == 0:
        # If no segments created, return the whole data padded/truncated
        piece = data[:length] if len(data) >= length else np.pad(data, (0, length - len(data)), mode='edge')
        rslt.append(piece)
    
    return np.array(rslt)

def prepare_time_series_data(dwell_times, length=50, stride=50):
    """
    Prepare single time series data for analysis.
    
    Parameters:
    -----------
    dwell_times : numpy array
        Dwell time series data
    length : int
        Length of subseries
    stride : int
        Stride for segmentation
    
    Returns:
    --------
    X : numpy array
        Prepared time series data (reshaped for shapelet learning)
    """
    # Segment the data
    segments = divide(dwell_times, length, stride)
    
    # Normalize each segment
    scaler = MinMaxScaler()
    segments_scaled = []
    for seg in segments:
        seg_reshaped = seg.reshape(-1, 1)
        seg_scaled = scaler.fit_transform(seg_reshaped).flatten()
        segments_scaled.append(seg_scaled)
    
    X = np.array(segments_scaled)
    
    # Reshape for shapelet learning: (n_samples, n_timestamps, n_features)
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X

"""# 4. Shapelet Analysis Functions"""

def get_available_gpus():
    """Check for available GPUs."""
    try:
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    except:
        return []

def train_shapelet_model(X_train, y_train, n_shapelets_per_size={20: 10}, 
                        max_iter=200, random_state=42):
    """
    Train a shapelet learning model.
    
    Parameters:
    -----------
    X_train : numpy array
        Training time series data
    y_train : numpy array
        Training labels
    n_shapelets_per_size : dict
        Number of shapelets per size
    max_iter : int
        Maximum iterations
    random_state : int
        Random seed
    
    Returns:
    --------
    model : LearningShapelets
        Trained model
    """
    print("Training shapelet model...")
    
    # Update optimizer to use learning_rate instead of deprecated lr
    try:
        optimizer = Adam(learning_rate=0.01)
    except:
        optimizer = Adam(lr=0.01)
    
    shp_clf = LearningShapelets(
        n_shapelets_per_size=n_shapelets_per_size,
        weight_regularizer=0.001,
        optimizer=optimizer,
        max_iter=max_iter,
        verbose=1,
        scale=False,
        random_state=random_state
    )
    
    shp_clf.fit(X_train, y_train)
    
    return shp_clf

def analyze_video_with_shapelets(video_path, reference_data=None, 
                                 length=50, stride=50, 
                                 n_shapelets_per_size={20: 10}):
    """
    Complete pipeline: Extract dwell times from video and analyze with shapelets.
    
    Parameters:
    -----------
    video_path : str
        Path to video file
    reference_data : numpy array, optional
        Reference time series for comparison/classification
    length : int
        Subseries length
    stride : int
        Stride for segmentation
    
    Returns:
    --------
    results : dict
        Dictionary containing analysis results
    """
    print("=" * 60)
    print("DWELL TIME ANALYSIS - SHAPELET METHOD")
    print("=" * 60)
    
    # Extract dwell times from video
    print("\n[Step 1/4] Extracting dwell times from video...")
    try:
        dwell_times, frame_data = extract_dwell_times_simple(video_path, method='intensity')
    except Exception as e:
        print(f"Error extracting dwell times: {e}")
        return None
    
    print(f"Extracted {len(dwell_times)} dwell time states")
    print(f"Mean dwell time: {np.mean(dwell_times):.2f} frames")
    print(f"Std dwell time: {np.std(dwell_times):.2f} frames")
    
    # Prepare time series data
    print("\n[Step 2/4] Preparing time series data...")
    X = prepare_time_series_data(dwell_times, length=length, stride=stride)
    print(f"Prepared {X.shape[0]} time series segments")
    
    # If reference data provided, perform classification
    if reference_data is not None:
        print("\n[Step 3/4] Training shapelet model with reference data...")
        # Prepare reference data
        X_ref = prepare_time_series_data(reference_data, length=length, stride=stride)
        
        # Create labels (assuming binary classification)
        y_ref = np.zeros(len(X_ref))
        y_video = np.ones(len(X))
        
        # Combine data
        X_combined = np.vstack([X_ref, X])
        y_combined = np.concatenate([y_ref, y_video])
        
        # Train model
        model = train_shapelet_model(X_combined, y_combined, 
                                    n_shapelets_per_size=n_shapelets_per_size)
        
        # Predict
        y_pred = model.predict(X)
        predictions = model.predict(X_combined)
        
        print(f"\nPredicted class distribution:")
        unique, counts = np.unique(y_pred, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {int(u)}: {c} samples ({100*c/len(y_pred):.1f}%)")
        
        # Calculate distances to shapelets
        distances = model.transform(X)
        
    else:
        print("\n[Step 3/4] Analyzing time series patterns...")
        # For single video analysis, we'll use unsupervised approach
        # Create synthetic labels for demonstration
        y = np.zeros(len(X))
        if len(X) > 1:
            y[len(X)//2:] = 1
        
        model = train_shapelet_model(X, y, n_shapelets_per_size=n_shapelets_per_size)
        distances = model.transform(X)
        predictions = model.predict(X)
    
    print("\n[Step 4/4] Generating visualizations...")
    
    results = {
        'dwell_times': dwell_times,
        'frame_data': frame_data,
        'time_series': X,
        'model': model,
        'distances': distances,
        'predictions': predictions if 'predictions' in locals() else None,
        'video_path': video_path
    }
    
    return results

"""# 5. Visualization Functions"""

def plot_visitor_times(visitor_times, output_dir="output"):
    """
    Create visualizations for visitor time analysis.
    
    Parameters:
    -----------
    visitor_times : list
        List of visitor time dictionaries
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(visitor_times) == 0:
        print("No visitors detected.")
        return
    
    # Extract data
    person_ids = [v['person_id'] for v in visitor_times]
    times_seconds = [v['time_seconds'] for v in visitor_times]
    times_minutes = [v['time_minutes'] for v in visitor_times]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Bar chart of time spent per visitor
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(person_ids)), times_seconds, color='#7FB4E7', edgecolor='black')
    ax1.set_xlabel('Visitor ID', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time Spent (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Time Spent by Each Visitor', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(person_ids)))
    ax1.set_xticklabels([f"Person {pid}" for pid in person_ids], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, times_seconds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s\n({times_minutes[i]:.2f}min)',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Pie chart of time distribution
    ax2 = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(person_ids)))
    wedges, texts, autotexts = ax2.pie(times_seconds, labels=[f"Person {pid}" for pid in person_ids],
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Time Distribution Among Visitors', fontsize=14, fontweight='bold')
    
    # 3. Time distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(times_seconds, bins=min(10, len(times_seconds)), color='#2674BD', 
            edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Time Spent (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Visitors', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Visitor Dwell Times', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    total_time = sum(times_seconds)
    avg_time = np.mean(times_seconds)
    median_time = np.median(times_seconds)
    min_time = min(times_seconds)
    max_time = max(times_seconds)
    std_time = np.std(times_seconds)
    
    stats_text = f"""
    VISITOR TIME ANALYSIS SUMMARY
    
    Total Visitors Detected: {len(visitor_times)}
    
    Time Statistics:
    • Total Time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)
    • Average Time: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)
    • Median Time: {median_time:.2f} seconds ({median_time/60:.2f} minutes)
    • Shortest Visit: {min_time:.2f} seconds ({min_time/60:.2f} minutes)
    • Longest Visit: {max_time:.2f} seconds ({max_time/60:.2f} minutes)
    • Std Deviation: {std_time:.2f} seconds
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visitor_time_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved visitor time analysis plot to {output_dir}/visitor_time_analysis.png")
    
    # Create detailed table
    fig2, ax = plt.subplots(figsize=(12, max(6, len(visitor_times) * 0.4)))
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Visitor ID', 'Time (seconds)', 'Time (minutes)', 'Time (formatted)', 
               'Frames Present', 'First Seen (frame)', 'Last Seen (frame)']
    
    for v in visitor_times:
        table_data.append([
            f"Person {v['person_id']}",
            f"{v['time_seconds']:.2f}",
            f"{v['time_minutes']:.2f}",
            v['time_formatted'],
            str(v['frames']),
            str(v['first_seen_frame']),
            str(v['last_seen_frame'])
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2674BD')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Detailed Visitor Time Report', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'visitor_time_table.png'), dpi=300, bbox_inches='tight')
    print(f"Saved visitor time table to {output_dir}/visitor_time_table.png")
    
    plt.close('all')

def plot_dwell_time_analysis(results, output_dir="output"):
    """
    Create comprehensive visualizations of the analysis results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_video_with_shapelets
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dwell_times = results['dwell_times']
    frame_data = results['frame_data']
    X = results['time_series']
    distances = results['distances']
    model = results['model']
    
    # 1. Dwell time distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(dwell_times, linewidth=2, color='#7FB4E7')
    plt.xlabel('State Index', fontsize=12)
    plt.ylabel('Dwell Time (frames)', fontsize=12)
    plt.title('Dwell Time Series', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.hist(dwell_times, bins=30, color='#2674BD', edgecolor='black', alpha=0.7)
    plt.xlabel('Dwell Time (frames)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Dwell Time Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Frame intensity/state data
    plt.subplot(2, 2, 3)
    plt.plot(frame_data[:min(1000, len(frame_data))], alpha=0.7, color='#7FB4E7')
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Normalized Intensity', fontsize=12)
    plt.title('Frame Intensity Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Shapelet visualization
    plt.subplot(2, 2, 4)
    if hasattr(model, 'shapelets_') and len(model.shapelets_) > 0:
        # Plot first few shapelets
        for i, shapelet in enumerate(model.shapelets_[:3]):
            shap_scaled = TimeSeriesScalerMinMax().fit_transform(
                shapelet.reshape(1, -1, 1)).flatten()
            plt.plot(shap_scaled, label=f'Shapelet {i+1}', linewidth=2)
        plt.xlabel('Time Index', fontsize=12)
        plt.ylabel('Normalized Value', fontsize=12)
        plt.title('Learned Shapelets', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No shapelets available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Learned Shapelets', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dwell_time_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved analysis plot to {output_dir}/dwell_time_analysis.png")
    
    # 4. Distance space visualization (if we have multiple shapelets)
    if distances.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(distances[:, 0], distances[:, 1], 
                   c=range(len(distances)), cmap='viridis', 
                   s=50, alpha=0.6, edgecolors='k')
        plt.xlabel('Distance to Shapelet 1', fontsize=14, fontweight='bold')
        plt.ylabel('Distance to Shapelet 2', fontsize=14, fontweight='bold')
        plt.title('Time Series in Shapelet Distance Space', fontsize=16, fontweight='bold')
        plt.colorbar(label='Sample Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shapelet_distance_space.png'), dpi=300, bbox_inches='tight')
        print(f"Saved distance space plot to {output_dir}/shapelet_distance_space.png")
    
    # 5. Time series segments with best matching shapelets
    if hasattr(model, 'shapelets_') and len(model.shapelets_) > 0:
        plt.figure(figsize=(14, 6))
        n_samples = min(3, len(X))
        for i in range(n_samples):
            plt.subplot(1, n_samples, i+1)
            ts = X[i].flatten()
            plt.plot(ts, label='Time Series', color='#7FB4E7', linewidth=2)
            
            # Find best matching shapelet
            sample_distances = distances[i]
            best_shapelet_idx = np.argmin(sample_distances)
            if best_shapelet_idx < len(model.shapelets_):
                shap = model.shapelets_[best_shapelet_idx]
                shap_scaled = TimeSeriesScalerMinMax().fit_transform(
                    shap.reshape(1, -1, 1)).flatten()
                # Align shapelet (simplified - in practice use locate())
                plt.plot(shap_scaled, label=f'Best Shapelet', 
                        color='#2674BD', linewidth=3, linestyle='--')
            
            plt.xlabel('Time Index', fontsize=10)
            plt.ylabel('Normalized Value', fontsize=10)
            plt.title(f'Sample {i+1}', fontsize=12)
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shapelet_matching.png'), dpi=300, bbox_inches='tight')
        print(f"Saved shapelet matching plot to {output_dir}/shapelet_matching.png")
    
    plt.close('all')
    print(f"\nAll visualizations saved to '{output_dir}' directory")

"""# 6. GUI Application for Video Upload and Analysis"""

class DwellTimeAnalysisGUI:
    """GUI application for video upload and dwell time analysis."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visitor Dwell Time Analysis")
        self.root.geometry("900x700")
        self.root.configure(bg='#2b2b2b')
        
        self.video_path = None
        self.use_camera = False
        self.camera_index = 0
        self.visitor_times = None
        self.processing = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Title
        title_frame = tk.Frame(self.root, bg='#2b2b2b')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame,
            text="VISITOR DWELL TIME ANALYSIS",
            font=('Arial', 20, 'bold'),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Upload a video to analyze visitor dwell times",
            font=('Arial', 12),
            bg='#2b2b2b',
            fg='#cccccc'
        )
        subtitle_label.pack(pady=5)
        
        detection_label = tk.Label(
            title_frame,
            text="✓ Human Detection Enabled - Only detects people, filters out other objects",
            font=('Arial', 10, 'italic'),
            bg='#2b2b2b',
            fg='#00ff88'
        )
        detection_label.pack(pady=2)
        
        # Upload section
        upload_frame = tk.Frame(self.root, bg='#2b2b2b')
        upload_frame.pack(pady=30)
        
        buttons_row = tk.Frame(upload_frame, bg='#2b2b2b')
        buttons_row.pack()
        
        self.upload_btn = tk.Button(
            buttons_row,
            text="📁 Upload Video File",
            font=('Arial', 14, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=30,
            pady=15,
            command=self.upload_video,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        self.camera_btn = tk.Button(
            buttons_row,
            text="📷 Use Live Camera",
            font=('Arial', 14, 'bold'),
            bg='#FF5722',
            fg='white',
            padx=30,
            pady=15,
            command=self.use_live_camera,
            cursor='hand2',
            relief=tk.RAISED,
            bd=3
        )
        self.camera_btn.pack(side=tk.LEFT, padx=10)
        
        # Video info display
        self.info_text = tk.Text(
            self.root,
            height=8,
            width=80,
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Courier', 10),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.info_text.pack(pady=20, padx=20)
        
        # Analysis button
        self.analyze_btn = tk.Button(
            self.root,
            text="▶ Start Analysis",
            font=('Arial', 14, 'bold'),
            bg='#2196F3',
            fg='white',
            padx=30,
            pady=15,
            command=self.start_analysis,
            cursor='hand2',
            state=tk.DISABLED,
            relief=tk.RAISED,
            bd=3
        )
        self.analyze_btn.pack(pady=10)
        
        # Progress bar
        self.progress_frame = tk.Frame(self.root, bg='#2b2b2b')
        self.progress_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="",
            font=('Arial', 10),
            bg='#2b2b2b',
            fg='#ffff00'
        )
        self.progress_label.pack()
        
        self.progress_bar = tk.ttk.Progressbar(
            self.progress_frame,
            mode='determinate',
            length=800
        )
        self.progress_bar.pack(pady=5)
        
        # Results section
        results_frame = tk.Frame(self.root, bg='#2b2b2b')
        results_frame.pack(pady=20, fill=tk.BOTH, expand=True, padx=20)
        
        results_label = tk.Label(
            results_frame,
            text="Analysis Results",
            font=('Arial', 14, 'bold'),
            bg='#2b2b2b',
            fg='#00ff00'
        )
        results_label.pack()
        
        # Results text area with scrollbar
        scroll_frame = tk.Frame(results_frame, bg='#2b2b2b')
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(
            scroll_frame,
            height=10,
            width=80,
            bg='#1e1e1e',
            fg='#00ff00',
            font=('Courier', 10),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        
        scrollbar = tk.Scrollbar(scroll_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons frame
        buttons_frame = tk.Frame(self.root, bg='#2b2b2b')
        buttons_frame.pack(pady=10)
        
        self.open_output_btn = tk.Button(
            buttons_frame,
            text="📊 View Charts",
            font=('Arial', 11),
            bg='#FF9800',
            fg='white',
            padx=15,
            pady=8,
            command=self.open_output_folder,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.open_output_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_csv_btn = tk.Button(
            buttons_frame,
            text="💾 Export CSV",
            font=('Arial', 11),
            bg='#9C27B0',
            fg='white',
            padx=15,
            pady=8,
            command=self.export_csv,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.export_csv_btn.pack(side=tk.LEFT, padx=5)
        
    def upload_video(self):
        """Open file dialog to select video."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.update_info(f"Video selected: {os.path.basename(file_path)}\n")
            self.analyze_btn.config(state=tk.NORMAL)
            
            # Get video info
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    
                    info = f"Resolution: {width}x{height}\n"
                    info += f"FPS: {fps:.2f}\n"
                    info += f"Duration: {duration:.1f} seconds\n"
                    info += f"Total frames: {total_frames}\n"
                    self.update_info(info)
                    cap.release()
            except Exception as e:
                self.update_info(f"Error reading video info: {e}\n")
    
    def use_live_camera(self):
        """Setup live camera input."""
        # Ask for camera index
        camera_dialog = tk.Toplevel(self.root)
        camera_dialog.title("Camera Setup")
        camera_dialog.geometry("400x200")
        camera_dialog.configure(bg='#2b2b2b')
        camera_dialog.transient(self.root)
        camera_dialog.grab_set()
        
        tk.Label(
            camera_dialog,
            text="Enter Camera Index:",
            font=('Arial', 12),
            bg='#2b2b2b',
            fg='white'
        ).pack(pady=20)
        
        camera_entry = tk.Entry(camera_dialog, font=('Arial', 12), width=10)
        camera_entry.insert(0, "0")
        camera_entry.pack(pady=10)
        
        def confirm_camera():
            try:
                self.camera_index = int(camera_entry.get())
                self.use_camera = True
                self.video_path = None
                self.update_info(f"Live camera selected (index: {self.camera_index})\n")
                self.update_info("Resolution: Will be detected from camera\n")
                self.update_info("FPS: Will be detected from camera\n")
                self.analyze_btn.config(state=tk.NORMAL)
                camera_dialog.destroy()
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter a valid camera index (integer)")
        
        tk.Button(
            camera_dialog,
            text="OK",
            font=('Arial', 12),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=5,
            command=confirm_camera
        ).pack(pady=20)
    
    def update_info(self, text):
        """Update info text area."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, text)
        self.info_text.see(tk.END)
        self.info_text.config(state=tk.DISABLED)
    
    def update_results(self, text):
        """Update results text area."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def clear_results(self):
        """Clear results text area."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def start_analysis(self):
        """Start the analysis process."""
        if not self.use_camera and (not self.video_path or not os.path.exists(self.video_path)):
            tk.messagebox.showerror("Error", "Please select a valid video file or camera first.")
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.analyze_btn.config(state=tk.DISABLED, text="Processing...")
        self.clear_results()
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Initializing...")
        
        # Run analysis in a separate thread to keep GUI responsive
        import threading
        thread = threading.Thread(target=self.run_analysis, daemon=True)
        thread.start()
    
    def run_analysis(self):
        """Run the analysis (called in separate thread)."""
        try:
            # Create output directory
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Update progress
            if self.use_camera:
                self.root.after(0, lambda: self.progress_label.config(text="Starting camera processing..."))
            else:
                self.root.after(0, lambda: self.progress_label.config(text="Starting video processing..."))
            self.root.after(0, lambda: self.progress_bar.config(value=10))
            
            # Generate output video path
            output_video_path = os.path.join(output_dir, "tracked_visitors.mp4")
            
            # Track visitors with live preview
            self.root.after(0, lambda: self.update_results("="*60 + "\n"))
            self.root.after(0, lambda: self.update_results("HUMAN DWELL TIME ANALYSIS\n"))
            self.root.after(0, lambda: self.update_results("="*60 + "\n\n"))
            if self.use_camera:
                self.root.after(0, lambda: self.update_results(f"Processing live camera (index: {self.camera_index}) with HUMAN DETECTION...\n"))
            else:
                self.root.after(0, lambda: self.update_results("Processing video with HUMAN DETECTION...\n"))
            if HAS_YOLO:
                self.root.after(0, lambda: self.update_results("Using YOLO (You Only Look Once) - State-of-the-art person detection\n"))
                self.root.after(0, lambda: self.update_results("✓ YOLO model loaded - Maximum accuracy for detecting all people\n"))
            else:
                self.root.after(0, lambda: self.update_results("Using HOG (Histogram of Oriented Gradients) for human detection\n"))
                self.root.after(0, lambda: self.update_results("Note: Install YOLO (pip install ultralytics) for better accuracy\n"))
            self.root.after(0, lambda: self.update_results("A window will open showing live processing.\n"))
            self.root.after(0, lambda: self.update_results("Press 'q' in the video window to finish early.\n\n"))
            
            self.root.after(0, lambda: self.progress_bar.config(value=20))
            
            # Run tracking
            if self.use_camera:
                visitor_times = track_visitors_in_camera(
                    camera_index=self.camera_index,
                    output_video_path=output_video_path,
                    show_preview=True
                )
            else:
                visitor_times = track_visitors_in_video(
                    self.video_path,
                    output_video_path=output_video_path,
                    show_preview=True
                )
            
            self.visitor_times = visitor_times
            
            self.root.after(0, lambda: self.progress_bar.config(value=60))
            self.root.after(0, lambda: self.progress_label.config(text="Generating visualizations..."))
            
            # Generate visualizations
            plot_visitor_times(visitor_times, output_dir=output_dir)
            
            self.root.after(0, lambda: self.progress_bar.config(value=80))
            self.root.after(0, lambda: self.progress_label.config(text="Finalizing results..."))
            
            # Display results
            self.display_results(visitor_times)
            
            # Save CSV
            df = pd.DataFrame(visitor_times)
            csv_path = os.path.join(output_dir, "visitor_times.csv")
            df.to_csv(csv_path, index=False)
            
            self.root.after(0, lambda: self.progress_bar.config(value=100))
            self.root.after(0, lambda: self.progress_label.config(text="Analysis Complete!"))
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL, text="▶ Start Analysis"))
            self.root.after(0, lambda: self.open_output_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.export_csv_btn.config(state=tk.NORMAL))
            
            self.root.after(0, lambda: self.update_results("\n" + "="*60 + "\n"))
            self.root.after(0, lambda: self.update_results("Analysis completed successfully!\n"))
            self.root.after(0, lambda: self.update_results(f"Results saved to: {output_dir}/\n"))
            self.root.after(0, lambda: self.update_results("="*60 + "\n"))
            
            # Show completion message
            self.root.after(0, lambda: tk.messagebox.showinfo(
                "Analysis Complete",
                f"Analysis completed!\n\n"
                f"Total visitors: {len(visitor_times)}\n"
                f"Results saved to: {output_dir}/\n\n"
                f"Click 'View Charts' to see visualizations."
            ))
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}\n"
            self.root.after(0, lambda: self.update_results(error_msg))
            self.root.after(0, lambda: self.progress_label.config(text="Error occurred"))
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL, text="▶ Start Analysis"))
            self.root.after(0, lambda: tk.messagebox.showerror("Error", f"Analysis failed:\n{str(e)}"))
            import traceback
            traceback.print_exc()
        finally:
            self.processing = False
    
    def display_results(self, visitor_times):
        """Display analysis results in the GUI."""
        if len(visitor_times) == 0:
            self.root.after(0, lambda: self.update_results("No visitors detected in the video.\n"))
            return
        
        # Sort by time spent
        visitor_times_sorted = sorted(visitor_times, key=lambda x: x['time_seconds'], reverse=True)
        
        # Calculate statistics
        total_time = sum(v['time_seconds'] for v in visitor_times)
        avg_time = np.mean([v['time_seconds'] for v in visitor_times])
        median_time = np.median([v['time_seconds'] for v in visitor_times])
        min_time = min(v['time_seconds'] for v in visitor_times)
        max_time = max(v['time_seconds'] for v in visitor_times)
        
        results = "\n" + "="*60 + "\n"
        results += "ANALYSIS RESULTS\n"
        results += "="*60 + "\n\n"
        
        results += f"Total Visitors Detected: {len(visitor_times)}\n\n"
        
        results += "STATISTICS:\n"
        results += "-"*60 + "\n"
        results += f"Total Time (all visitors): {total_time/60:.2f} minutes ({total_time:.1f} seconds)\n"
        results += f"Average Time per Visitor: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)\n"
        results += f"Median Time: {median_time:.2f} seconds ({median_time/60:.2f} minutes)\n"
        results += f"Shortest Visit: {min_time:.2f} seconds ({min_time/60:.2f} minutes)\n"
        results += f"Longest Visit: {max_time:.2f} seconds ({max_time/60:.2f} minutes)\n\n"
        
        results += "TIME SPENT BY EACH VISITOR:\n"
        results += "-"*60 + "\n"
        results += f"{'Visitor ID':<12} {'Time':<15} {'Frames':<10} {'Status'}\n"
        results += "-"*60 + "\n"
        
        for v in visitor_times_sorted:
            status = "Active" if v['time_seconds'] > avg_time else "Normal"
            results += f"Visitor #{v['person_id']:<5} {v['time_formatted']:<15} {v['frames']:<10} {status}\n"
        
        results += "\n" + "="*60 + "\n"
        
        self.root.after(0, lambda: self.update_results(results))
    
    def open_output_folder(self):
        """Open the output folder."""
        output_dir = os.path.abspath("output")
        if os.path.exists(output_dir):
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                os.system(f"open {output_dir}")
            else:
                os.system(f"xdg-open {output_dir}")
        else:
            tk.messagebox.showwarning("Warning", "Output folder not found.")
    
    def export_csv(self):
        """Export results to CSV."""
        if self.visitor_times is None or len(self.visitor_times) == 0:
            tk.messagebox.showwarning("Warning", "No data to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save CSV File",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.visitor_times)
                df.to_csv(file_path, index=False)
                tk.messagebox.showinfo("Success", f"Data exported to:\n{file_path}")
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to export CSV:\n{str(e)}")
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

"""# 7. Main Execution"""

def select_video_file():
    """Open file dialog to select video file."""
    if HAS_TKINTER:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return file_path
    else:
        return None

def main():
    """Main execution function - AUTO MODE: Automatically opens camera and starts detection."""
    # AUTO MODE: Automatically use camera and start detection
    use_camera = True
    camera_index = 0
    analysis_type = "1"  # Default to visitor tracking
    video_path = None
    
    # Check if user wants to use video file instead
    if len(sys.argv) > 1:
        arg1 = sys.argv[1].lower()
        if arg1 in ['--gui', '-g', '--window']:
            # Launch GUI if requested
            if HAS_TKINTER:
                app = DwellTimeAnalysisGUI()
                app.run()
                return
        elif arg1 in ['--video', '-v', '--file']:
            # Use video file mode
            use_camera = False
            if len(sys.argv) > 2:
                video_path = sys.argv[2]
            else:
                if HAS_TKINTER:
                    video_path = select_video_file()
                    if not video_path:
                        print("No file selected. Using camera instead.")
                        use_camera = True
                else:
                    video_path = input("\nEnter path to video file: ").strip().strip('"')
        elif arg1 in ['--camera', '-cam', '--live']:
            use_camera = True
            if len(sys.argv) > 2:
                try:
                    camera_index = int(sys.argv[2])
                except ValueError:
                    camera_index = 0
        elif arg1 not in ['--gui', '-g', '--window', '--cli', '-c', '--command', '--camera', '-cam', '--live']:
            # Assume it's a video file path
            if os.path.exists(arg1):
                use_camera = False
                video_path = arg1
    
    # CLI mode
    print("=" * 60)
    print("VISITOR DWELL TIME ANALYSIS - AUTO MODE")
    print("=" * 60)
    
    if not use_camera:
        if not video_path or not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            print("Switching to camera mode...")
            use_camera = True
            camera_index = 0
        else:
            print(f"\nVideo file: {video_path}")
    else:
        print(f"\nAUTO STARTING LIVE CAMERA (index: {camera_index})")
        print("Detection will start automatically...")
    
    try:
        if analysis_type == "1":
            # Visitor tracking analysis - AUTO MODE
            print("\n" + "=" * 60)
            print("VISITOR TRACKING AND TIME ANALYSIS - AUTO MODE")
            print("=" * 60)
            if use_camera:
                print("✓ Camera opened automatically")
                print("✓ Detection started automatically")
                print("✓ Tracking people and calculating dwell time")
                print("✓ Duplicate prevention enabled")
            else:
                print("Live processing will start - you'll see the video with tracking overlay")
            print("Press 'q' to quit, 'p' to pause/resume during processing\n")
            
            # AUTO MODE: Automatically save annotated video
            save_video = True
            output_video_path = None
            # Check for output path in command line arguments
            arg_idx = 3 if use_camera else 3
            if len(sys.argv) > arg_idx:
                output_video_path = sys.argv[arg_idx]
            else:
                # Auto-save to output directory
                os.makedirs("output", exist_ok=True)
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                output_video_path = os.path.join("output", f"tracked_visitors_{timestamp}.mp4")
            
            # Track visitors with LIVE preview (always enabled)
            if use_camera:
                visitor_times = track_visitors_in_camera(
                    camera_index=camera_index,
                    output_video_path=output_video_path if save_video else None,
                    show_preview=True  # Always show live preview
                )
            else:
                visitor_times = track_visitors_in_video(
                    video_path,
                    output_video_path=output_video_path if save_video else None,
                    show_preview=True  # Always show live preview
                )
            
            if len(visitor_times) == 0:
                print("\nNo visitors detected in the video.")
                print("Try adjusting detection parameters or check video quality.")
                return
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            plot_visitor_times(visitor_times, output_dir="output")
            
            # Print summary
            print("\n" + "=" * 60)
            print("VISITOR TIME ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"\nTotal Visitors Detected: {len(visitor_times)}\n")
            
            total_time = sum(v['time_seconds'] for v in visitor_times)
            avg_time = np.mean([v['time_seconds'] for v in visitor_times])
            
            print(f"Total Time (all visitors): {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
            print(f"Average Time per Visitor: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)\n")
            
            print("Time Spent by Each Visitor:")
            print("-" * 60)
            for v in visitor_times:
                print(f"Person {v['person_id']:2d}: {v['time_formatted']:>8s} "
                      f"({v['time_seconds']:6.2f}s, {v['frames']:4d} frames)")
            print("=" * 60)
            
            # Save results to CSV
            df = pd.DataFrame(visitor_times)
            csv_path = os.path.join("output", "visitor_times.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to CSV: {csv_path}")
            if save_video:
                print(f"Annotated video saved to: {output_video_path}")
            print("\nAnalysis complete! Check the 'output' directory for visualizations.")
            
        else:
            # Original shapelet analysis
            print("\n" + "=" * 60)
            print("SHAPELET-BASED DWELL TIME ANALYSIS")
            print("=" * 60)
            
            # Check for GPU
            gpus = get_available_gpus()
            if gpus:
                print(f"GPU available: {', '.join(gpus)}")
            else:
                print("No GPU detected. Using CPU (may be slower).")
            
            # Perform analysis
            results = analyze_video_with_shapelets(
                video_path,
                reference_data=None,
                length=50,
                stride=50,
                n_shapelets_per_size={20: 10}
            )
            
            if results is None:
                print("Analysis failed.")
                return
            
            # Generate visualizations
            plot_dwell_time_analysis(results, output_dir="output")
            
            # Print summary statistics
            print("\n" + "=" * 60)
            print("ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"Total dwell time states: {len(results['dwell_times'])}")
            print(f"Mean dwell time: {np.mean(results['dwell_times']):.2f} frames")
            print(f"Median dwell time: {np.median(results['dwell_times']):.2f} frames")
            print(f"Std deviation: {np.std(results['dwell_times']):.2f} frames")
            print(f"Min dwell time: {np.min(results['dwell_times'])} frames")
            print(f"Max dwell time: {np.max(results['dwell_times'])} frames")
            print(f"Time series segments: {results['time_series'].shape[0]}")
            print(f"Shapelets learned: {len(results['model'].shapelets_) if hasattr(results['model'], 'shapelets_') else 0}")
            print("=" * 60)
            print("\nAnalysis complete! Check the 'output' directory for visualizations.")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # AUTO MODE: Automatically start camera detection
    # Use --gui flag to launch GUI instead
    if len(sys.argv) > 1 and sys.argv[1] in ['--gui', '-g', '--window']:
        if HAS_TKINTER:
            app = DwellTimeAnalysisGUI()
            app.run()
        else:
            print("GUI not available. Starting auto camera mode...")
            main()
    else:
        # AUTO MODE: Start camera automatically
        main()
