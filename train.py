import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance
import torch.multiprocessing as mp
from facenet_pytorch import InceptionResnetV1  # Import pre-trained face model
from model import SiameseNetwork  # Ensure SiameseNetwork is imported

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model correctly
model = SiameseNetwork(backbone="resnet101").to(device)

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Function to enhance image brightness
def adjust_brightness(image, factor=1.5):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

# Load Pre-trained FaceNet Model for Feature Extraction
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Custom Dataset
class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.GaussianBlur(3),
            transforms.ToTensor()
        ])
        self.people = os.listdir(root_dir)
        self.image_paths = {
            person: [os.path.join(root_dir, person, img) for img in os.listdir(os.path.join(root_dir, person))]
            for person in self.people
        }
        self.pairs, self.labels = self.create_pairs()

    def extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = facenet(image).cpu().numpy()
        return embedding

    def create_pairs(self):
        pairs, labels = [], []
        embeddings = {}

        # Compute embeddings for all images
        for person in self.people:
            embeddings[person] = [self.extract_features(img) for img in self.image_paths[person]]

        for person in self.people:
            images = self.image_paths[person]
            for i in range(len(images) - 1):
                pairs.append((images[i], images[i + 1]))
                labels.append(1)

            # Hard Negative Mining using Feature Similarity
            person_embedding = np.mean(embeddings[person], axis=0)
            distances = {
                other: np.linalg.norm(person_embedding - np.mean(embeddings[other], axis=0))
                for other in self.people if other != person
            }
            hard_negatives = sorted(distances, key=distances.get)[:3]  # Select 3 closest negatives

            for neg_person in hard_negatives:
                neg_image = random.choice(self.image_paths[neg_person])
                pairs.append((random.choice(images), neg_image))
                labels.append(0)

        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs, labels = zip(*combined)
        return pairs, labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1 = Image.open(self.pairs[idx][0]).convert("RGB")
        img2 = Image.open(self.pairs[idx][1]).convert("RGB")
        img1 = adjust_brightness(img1)
        img2 = adjust_brightness(img2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img1, img2, label

# Ensure main guard for Windows compatibility
if __name__ == "__main__":
    mp.freeze_support()

    # Training Setup
    dataset = FaceDataset("data")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    criterion = ContrastiveLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    for epoch in range(30):  # Increased epochs for better convergence
        total_loss = 0
        model.train()

        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            with torch.cuda.amp.autocast():
                output1 = model(img1)
                output2 = model(img2)
                output1 = nn.functional.normalize(output1, p=2, dim=1)  # L2 normalization
                output2 = nn.functional.normalize(output2, p=2, dim=1)
                loss = criterion(output1, output2, label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    # Save Model
    torch.save(model.state_dict(), "siamese_model.pth")
    print("Model saved!")
