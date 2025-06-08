import cv2
import os
import time
from mtcnn import MTCNN

# Initialize webcam
video = cv2.VideoCapture(0)
detector = MTCNN()

# Create dataset directory
data_path = "data"
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Ask for the person's name
name = input("Enter your name: ").strip()
person_path = os.path.join(data_path, name)

if not os.path.exists(person_path):
    os.makedirs(person_path)

# Capture face images
count = 0
max_images = 100  # More images improve accuracy

print(f"Capturing {max_images} face images for {name}. Please look straight, blink, and turn slightly.")

while count < max_images:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']

        
        margin = int(0.2 * w)
        x, y = max(0, x - margin), max(0, y - margin)
        w, h = min(frame.shape[1] - x, w + 2 * margin), min(frame.shape[0] - y, h + 2 * margin)

        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))  # Resize for consistency
        cv2.imwrite(f"{person_path}/{count}.jpg", face_img)
        count += 1

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Captured {count}/{max_images}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Registration", frame)

    time.sleep(0.2)  # Small delay to avoid capturing duplicate frames

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
print(f"✅ Face images saved for {name}")
