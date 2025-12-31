# -*- coding: utf-8 -*-
"""Face Recognition with Live Camera and Attendance System"""

# Step 1. Import required libraries
import numpy as np
import pandas as pd
import cv2
import os
from datetime import datetime

# Step 2. Initialize face detector
try:
    # Check if CascadeClassifier exists
    if not hasattr(cv2, 'CascadeClassifier'):
        raise AttributeError("CascadeClassifier not found in cv2")
    
    # Check if cv2.data exists
    if not hasattr(cv2, 'data') or not hasattr(cv2.data, 'haarcascades'):
        raise AttributeError("cv2.data.haarcascades not found")
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise ValueError("CascadeClassifier failed to load")
        
except (AttributeError, ValueError) as e:
    print("\n" + "="*60)
    print("ERROR: OpenCV face detection not available")
    print("="*60)
    print(f"Error details: {str(e)}")
    print("\nYour OpenCV installation appears to be incomplete or corrupted.")
    print("\nPlease reinstall OpenCV with the following commands:")
    print("  pip uninstall opencv-python opencv-python-headless -y")
    print("  pip install opencv-python")
    print("\nIf you're using conda:")
    print("  conda uninstall opencv -y")
    print("  conda install -c conda-forge opencv")
    print("\nAfter reinstalling, restart your Python environment.")
    print("="*60)
    exit(1)

# Step 3. Load images from folder and extract face templates
def load_face_templates(folder_path):
    """Load images from folder, extract faces, and create templates for matching"""
    known_faces = []
    known_face_names = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return [], []
    
    # Get all image files from folder
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"No images found in {folder_path}")
        return [], []
    
    print(f"Found {len(image_files)} images. Processing...")
    
    # Standard face size for comparison
    FACE_SIZE = (100, 100)
    
    for filename in sorted(image_files):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not load {filename}")
            continue
        
        # Detect faces in the image
        detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)
        
        if len(detected_faces) == 0:
            print(f"Warning: No face detected in {filename}")
            continue
        
        # Use the first detected face
        (x, y, w, h) = detected_faces[0]
        face_roi = img[y:y+h, x:x+w]
        
        # Resize to standard size for comparison
        face_resized = cv2.resize(face_roi, FACE_SIZE)
        
        # Extract name from filename (without extension)
        person_name = os.path.splitext(filename)[0]
        
        # Add face template and name
        known_faces.append(face_resized)
        known_face_names.append(person_name)
        
        print(f"  - Loaded: {person_name}")
    
    if len(known_faces) == 0:
        print("Error: No faces found in any images")
        return [], []
    
    print(f"\nLoaded {len(known_face_names)} face templates. Ready for recognition.")
    return known_faces, known_face_names

# Step 4. Load reference images and create templates
script_folder = os.path.dirname(os.path.abspath(__file__))
known_faces, known_face_names = load_face_templates(script_folder)

if len(known_faces) == 0:
    print("\nError: Could not load any face templates. Please ensure images with faces are in the folder.")
    exit(1)

# Step 5. Face matching function using histogram comparison
def compare_faces(face1, face2):
    """Compare two faces using histogram correlation"""
    # Calculate histograms
    hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
    
    # Compare histograms using correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return correlation

# Step 6. Attendance tracking
attendance_file = 'attendance.csv'

def mark_attendance(person_name):
    """Mark attendance in CSV file using person name"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.now().strftime("%Y-%m-%d")
    
    # Check if attendance file exists
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        # Check if person already marked today
        today_records = df[(df['Name'] == person_name) & (df['Date'] == date)]
        if len(today_records) == 0:
            new_record = pd.DataFrame({
                'Name': [person_name],
                'Date': [date],
                'Time': [timestamp]
            })
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(attendance_file, index=False)
            return True
        return False
    else:
        # Create new attendance file
        df = pd.DataFrame({
            'Name': [person_name],
            'Date': [date],
            'Time': [timestamp]
        })
        df.to_csv(attendance_file, index=False)
        return True

# Step 7. Real-time face recognition with camera
def recognize_faces():
    """Main function for real-time face recognition"""
    # Try GUI mode by default (will fall back to headless if it fails)
    gui_available = True  # Default to trying GUI mode
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened. Press 'q' in the camera window to quit (or Ctrl+C if window doesn't appear).")
    
    # Track recognized faces to avoid duplicate attendance
    last_attendance_time = {}
    attendance_cooldown = 5  # seconds between attendance marks for same person
    
    # Face matching threshold (correlation, higher is better, typically > 0.7)
    match_threshold = 0.7
    FACE_SIZE = (100, 100)  # Same size as stored templates
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Resize to standard size for comparison
                face_resized = cv2.resize(face_roi, FACE_SIZE)
                
                # Compare with all known faces
                best_match_index = -1
                best_match_score = 0
                
                for i, known_face in enumerate(known_faces):
                    score = compare_faces(face_resized, known_face)
                    if score > best_match_score:
                        best_match_score = score
                        best_match_index = i
                
                # Check if match is good enough
                if best_match_index >= 0 and best_match_score >= match_threshold:
                    person_name = known_face_names[best_match_index]
                    confidence = best_match_score * 100
                    
                    # Mark attendance (with cooldown)
                    current_time = datetime.now()
                    if person_name not in last_attendance_time or \
                       (current_time - last_attendance_time[person_name]).seconds > attendance_cooldown:
                        if mark_attendance(person_name):
                            print(f"[{current_time.strftime('%H:%M:%S')}] ✓ Recognized: {person_name} (Confidence: {confidence:.1f}%) - Attendance marked")
                        else:
                            print(f"[{current_time.strftime('%H:%M:%S')}] ✓ Recognized: {person_name} (Confidence: {confidence:.1f}%)")
                        last_attendance_time[person_name] = current_time
                    
                    # Draw on frame if GUI is available
                    if gui_available:
                        color = (0, 255, 0) if confidence > 85 else (0, 165, 255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        label_text = f"{person_name} ({confidence:.1f}%)"
                        cv2.putText(frame, label_text, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    # Unknown person
                    if gui_available:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display attendance info on frame if GUI is available
            if gui_available:
                try:
                    if os.path.exists(attendance_file):
                        df = pd.read_csv(attendance_file)
                        today = datetime.now().strftime("%Y-%m-%d")
                        today_count = len(df[df['Date'] == today])
                        cv2.putText(frame, f"Today's Attendance: {today_count}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display frame
                    cv2.imshow('Face Recognition - Attendance System', frame)
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    # GUI failed during use, switch to headless mode
                    gui_available = False
                    print("\nGUI display failed. Switching to headless mode...")
                    print("Face recognition continues - attendance will still be marked.\n")
            else:
                # In headless mode, just process frames
                # Print status every 60 frames (~2 seconds at 30fps) if faces detected
                if frame_count % 60 == 0 and len(faces) > 0:
                    print(f"Processing... Detected {len(faces)} face(s)")
    
    except KeyboardInterrupt:
        print("\nStopping face recognition...")
    finally:
        # Release resources
        cap.release()
        if gui_available:
            cv2.destroyAllWindows()
        print("Camera closed.")

# Step 8. Run the face recognition system
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Face Recognition and Attendance System")
    print("="*50)
    print("\nInstructions:")
    print("- Make sure your face is clearly visible")
    print("- If GUI is available: Press 'q' to quit")
    print("- If no GUI: Press Ctrl+C to stop")
    print("- Attendance is automatically marked when recognized")
    print("- Check attendance.csv for attendance records")
    print("="*50 + "\n")
    
    recognize_faces()
