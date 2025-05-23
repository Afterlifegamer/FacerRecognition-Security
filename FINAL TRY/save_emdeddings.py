import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import SiameseNetwork

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork(backbone="resnet101").to(device)
model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
model.eval()

# Load images and compute embeddings
dataset_path = "data"
people = os.listdir(dataset_path)
transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
embeddings = {}

for person in people:
    img_path = os.path.join(dataset_path, person, os.listdir(os.path.join(dataset_path, person))[0])
    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)
    # Ensure image is in 3 channels
    img = img.repeat(1, 3, 1, 1)  # Convert (1, 100, 100) -> (3, 100, 100)

# Get embedding
    embeddings[person] = model.get_embedding(img).detach().cpu().numpy()


# Save embeddings to file
embedding = model.get_embedding(img).detach().cpu().numpy()
embedding /= np.linalg.norm(embedding)  # ✅ Normalize before saving


np.save("face_embeddings.npy", embeddings)
print("✅ Face embeddings saved successfully!")
