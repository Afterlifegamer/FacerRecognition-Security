import os
import torch
import smtplib
import numpy as np
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import cv2
from email.message import EmailMessage
from model import SiameseNetwork
import time
# Email setup
EMAIL_SENDER = "22cs009@mgits.ac.in"  
EMAIL_PASSWORD = "ebvn mmfd iwmx blrq"  
EMAIL_RECEIVER = "abhiramsabu206@gmail.com"  

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork(backbone="resnet101").to(device)
model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
model.eval()

# Load stored embeddings
stored_embeddings = np.load("face_embeddings.npy", allow_pickle=True).item()
for person in stored_embeddings:
    stored_embeddings[person] = stored_embeddings[person] / np.linalg.norm(stored_embeddings[person])

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=False, device=device)

# Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

last_alert_time = 0  # Track last alert timestamp
ALERT_INTERVAL = 60  # Minimum time (seconds) between alerts

def send_email_alert(image_path):
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time >= ALERT_INTERVAL:
        msg = EmailMessage()
        msg["Subject"] = "🚨 Unknown Person Detected!"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content("An unknown person was detected. See the attached image.")
        
        with open(image_path, "rb") as img_file:
            msg.add_attachment(img_file.read(), maintype="image", subtype="jpeg", filename="unknown_person.jpg")
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        last_alert_time = current_time  # Update last alert timestamp
        print("📨 Alert email sent!")
    else:
        print("⏳ Alert skipped to prevent spam (cooldown active)")

# Function to capture screenshot
def capture_screenshot(frame, filename="unknown_person.jpg"):
    cv2.imwrite(filename, frame)
    print("📸 Screenshot captured: ", filename)
    return filename

# Webcam Setup
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for MTCNN
    img_pil = Image.fromarray(rgb_frame)
    boxes, _ = mtcnn.detect(img_pil)

    if boxes is not None:
        for box in boxes:
            x, y, x2, y2 = map(int, box)

            # Slightly expand face bounding box
            margin = int(0.2 * (x2 - x))
            x, y = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(frame.shape[1], x2 + margin), min(frame.shape[0], y2 + margin)

            face = img_pil.crop((x, y, x2, y2))
            face_tensor = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                live_embedding = model.get_embedding(face_tensor)
                live_embedding = torch.nn.functional.normalize(live_embedding, p=2, dim=1).cpu().numpy().flatten()

            max_similarity = -1
            best_match = "Unknown"
            threshold = 0.9  

            for person, stored_embedding in stored_embeddings.items():
                similarity = np.dot(live_embedding, stored_embedding)  # Cosine similarity
                if similarity > max_similarity: 
                    max_similarity = similarity
                    best_match = person if similarity > threshold else "Unknown"

            print(f"Detected: {best_match}, Similarity: {max_similarity:.4f}")
            print(f"Live Embedding: {live_embedding[:5]}")  # Print first few values
            print(f"Stored Embedding: {stored_embedding[:5]}")
            print(f"Similarity: {similarity}")


            # Draw bounding box & name
            color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, best_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Send email alert if unknown
            if best_match == "Unknown":
                screenshot_path = capture_screenshot(frame)
                send_email_alert(screenshot_path)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
