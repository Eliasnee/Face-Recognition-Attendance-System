import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import os
from pathlib import Path
import numpy as np
import cv2
import random

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2')

        # Freeze more layers to prevent overfitting
        for param in list(self.facenet.parameters())[:-5]:  # Freeze more layers
            param.requires_grad = False

        # Improved additional layers with stronger regularization
        self.additional = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, momentum=0.01),  # Lower momentum for better stability
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, momentum=0.01)
        )

    def forward_one(self, x):
        try:
            x = self.facenet(x)
            x = self.additional(x)
            x = F.normalize(x, p=2, dim=1)
            return x
        except RuntimeError as e:
            print(f"Error in forward_one: {str(e)}")
            print(f"Input shape: {x.shape}")
            raise e

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

class SiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None, unknown_ratio=0.2):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.unknown_ratio = unknown_ratio
        
        # Get all classes including Unknown
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_images = {}
        
        # Group images by class
        for cls in self.classes:
            cls_path = self.root_dir / cls
            self.class_to_images[cls] = [
                str(f) for f in cls_path.glob('*.jpg')
            ]
        
        # Special handling for Unknown class
        self.has_unknown = "Unknown" in self.classes
        if self.has_unknown:
            print(f"Found {len(self.class_to_images['Unknown'])} unknown face images")
    
    def __len__(self):
        return 2000  # Number of pairs per epoch
    
    def __getitem__(self, idx):
        # Handle unknown faces with probability unknown_ratio
        if self.has_unknown and random.random() < self.unknown_ratio:
            # Create pairs with unknown faces
            if random.random() < 0.5:
                # Unknown vs Known (should be different)
                img1_path = random.choice(self.class_to_images["Unknown"])
                cls = random.choice([c for c in self.classes if c != "Unknown"])
                img2_path = random.choice(self.class_to_images[cls])
                should_get_same_class = False
            else:
                # Unknown vs Unknown (should be different)
                img1_path, img2_path = random.sample(self.class_to_images["Unknown"], 2)
                should_get_same_class = False
        else:
            # Original positive/negative pair logic
            should_get_same_class = random.random() < 0.5
            valid_classes = [c for c in self.classes if c != "Unknown"]
            
            if should_get_same_class:
                cls = random.choice(valid_classes)
                if len(self.class_to_images[cls]) < 2:
                    should_get_same_class = False
                else:
                    img1_path, img2_path = random.sample(self.class_to_images[cls], 2)
            
            if not should_get_same_class:
                cls1, cls2 = random.sample(valid_classes, 2)
                img1_path = random.choice(self.class_to_images[cls1])
                img2_path = random.choice(self.class_to_images[cls2])
        
        # Load and transform images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.FloatTensor([should_get_same_class])

class OnlineContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, confidence_penalty=0.1):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.confidence_penalty = confidence_penalty
    
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        cosine_similarity = F.cosine_similarity(output1, output2)
        
        # Convert label to correct shape and type
        pos_mask = label.bool().squeeze()
        neg_mask = ~pos_mask
        
        # Handle positive pairs
        pos_loss = 0
        if pos_mask.any():
            pos_dist = euclidean_distance[pos_mask]
            pos_loss = torch.mean(pos_dist)
            
            # Add penalty for overconfident positive predictions
            pos_sim = cosine_similarity[pos_mask]
            overconfidence = torch.mean(F.relu(pos_sim - 0.9))
            pos_loss += self.confidence_penalty * overconfidence
        
        # Handle negative pairs
        neg_loss = 0
        if neg_mask.any():
            neg_dist = euclidean_distance[neg_mask]
            hard_negatives = neg_dist[neg_dist < self.margin]
            if len(hard_negatives) > 0:
                neg_loss = torch.mean(F.relu(self.margin - hard_negatives))
                
                # Add penalty for overconfident negative predictions
                neg_sim = cosine_similarity[neg_mask]
                overconfidence = torch.mean(F.relu(0.3 - neg_sim))
                neg_loss += self.confidence_penalty * overconfidence
        
        return pos_loss + neg_loss

def train_siamese_network(data_dir, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the network
    net = SiameseNetwork().to(device)
    
    # Define the loss function and optimizer
    criterion = OnlineContrastiveLoss(margin=1.0)
    
    pretrained_params = list(net.facenet.parameters())[-10:]
    new_params = list(net.additional.parameters())
    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': 0.0001, 'weight_decay': 0.01},
        {'params': new_params, 'lr': 0.0003, 'weight_decay': 0.01}
    ])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Define transforms - enhanced augmentation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Larger size for random cropping
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.GaussianBlur(kernel_size=3),
        ], p=0.7),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomCrop(224),  # Crop to final size
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])

    print(f"Loading dataset from: {data_dir}")
    train_dataset = SiameseDataset(data_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        batch_count = 0
        
        try:
            for i, data in enumerate(train_loader):
                img1, img2, label = data
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                
                optimizer.zero_grad()
                output1, output2 = net(img1, img2)
                loss = criterion(output1, output2, label)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                batch_count += 1
                
                if i % 10 == 0:
                    print(f"Batch {i}, Current loss: {loss.item():.4f}")
            
            epoch_loss = running_loss / batch_count
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
            
            scheduler.step(epoch_loss)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                torch.save(net.state_dict(), "best_siamese_model.pth")
                print(f"Saved model with loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
                    
        except Exception as e:
            print(f"Error during training: {str(e)}")
            torch.save(net.state_dict(), "backup_model.pth")
            raise e

    print("Training completed!")
    return net

def save_model(model, path):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    data_dir = "reference_faces"  # Using local reference_faces directory
    model = train_siamese_network(data_dir)
    save_model(model, "siamese_face_model.pth")
