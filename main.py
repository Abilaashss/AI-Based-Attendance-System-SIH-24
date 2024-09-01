import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import os

def load_known_faces(directory):
    known_faces = []
    known_names = []
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            try:
                image = face_recognition.load_image_file(os.path.join(directory, filename))
                encoding = face_recognition.face_encodings(image)[0]
                known_faces.append(encoding)
                known_names.append(os.path.splitext(filename)[0])
                print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    print(f"Loaded {len(known_faces)} known faces")
    return known_faces, known_names

def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not capture frame.")
        return None
    print(f"Captured frame shape: {frame.shape}")
    return frame

def recognize_face(frame, known_faces, known_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"RGB frame shape: {rgb_frame.shape}")
    
    face_locations = face_recognition.face_locations(rgb_frame)
    print(f"Detected {len(face_locations)} faces")
    
    if not face_locations:
        print("No faces detected in the frame.")
        return None

    try:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(f"Generated {len(face_encodings)} face encodings")
    except Exception as e:
        print(f"Error during face encoding: {e}")
        print(f"Face locations: {face_locations}")
        return None

    for face_encoding in face_encodings:
        try:
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            print(f"Face comparison results: {matches}")
            
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_names[best_match_index]
                print(f"Best match: {name} (distance: {face_distances[best_match_index]})")
                return name
            else:
                print(f"No match found. Closest distance: {face_distances[best_match_index]}")
        except Exception as e:
            print(f"Error during face comparison: {e}")
    
    return None

def mark_attendance(name):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")
    
    filename = f"{date_string}.csv"
    
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Time"])
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, time_string])
    
    print(f"Marked attendance for {name} at {time_string}")

def main():
    known_faces_dir = "known_faces"
    
    if not os.path.exists(known_faces_dir):
        print(f"Error: Directory '{known_faces_dir}' does not exist.")
        return
    
    known_faces, known_names = load_known_faces(known_faces_dir)
    
    if not known_faces:
        print("Error: No known faces loaded. Please check the 'known_faces' directory.")
        return
    
    frame = capture_frame()
    
    if frame is None:
        print("Error: Failed to capture frame from camera.")
        return

    recognized_name = recognize_face(frame, known_faces, known_names)
    
    if recognized_name:
        print(f"Recognized: {recognized_name}")
        mark_attendance(recognized_name)
        print("Attendance marked successfully!")
    else:
        print("No face recognized or error occurred during recognition.")

if __name__ == "__main__":
    main()