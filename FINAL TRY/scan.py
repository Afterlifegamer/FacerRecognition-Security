import cv2
import torch
import numpy as np
from model import SiameseNetwork
from PIL import Image
from torchvision import transforms

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork(backbone="resnet101").to(device)
model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
model.eval()

# Load stored embeddings
stored_embeddings = np.load("face_embeddings.npy", allow_pickle=True).item()

# Webcam Setup
video = cv2.VideoCapture(0)
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

def preprocess_face(face):
    image = Image.fromarray(face).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_tensor = preprocess_face(face)
        
        with torch.no_grad():
            live_embedding = model.get_embedding(face_tensor).cpu().numpy().flatten()
        
        min_dist = float("inf")
        best_match = "Unknown"
        
        for person, stored_embedding in stored_embeddings.items():
            dist = np.linalg.norm(live_embedding - stored_embedding)
            if dist < min_dist and dist < 0.5:  # Adjusted threshold
                min_dist = dist
                best_match = person
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, best_match, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
