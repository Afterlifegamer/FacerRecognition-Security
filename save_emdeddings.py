import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from model import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SiameseNetwork(backbone="resnet101").to(device)
model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
model.eval()

# Initialize MTCNN
mtcnn = MTCNN(keep_all=False, device=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Directory of stored faces
data_path = "data"
stored_embeddings = {}

for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)
    embeddings = []

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = Image.open(image_path).convert("RGB")
        face_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.get_embedding(face_tensor).cpu().numpy().flatten()
            embeddings.append(embedding)

    if embeddings:
        mean_embedding = np.mean(embeddings, axis=0)
        stored_embeddings[person] = mean_embedding / np.linalg.norm(mean_embedding)  # Normalize


np.save("face_embeddings.npy", stored_embeddings)
print("✅ Stored embeddings updated!")
