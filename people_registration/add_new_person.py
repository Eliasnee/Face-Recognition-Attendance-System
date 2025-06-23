import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import random
import os
import numpy as np
from collections import defaultdict

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        
        # More gradual unfreezing - freeze earlier layers more aggressively
        for param in list(self.facenet.parameters())[:-10]:
            param.requires_grad = False
            
        self.additional = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim, momentum=0.01)
        )

    def forward_one(self, x):
        x = self.facenet(x)
        x = self.additional(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, input1, input2):
        return self.forward_one(input1), self.forward_one(input2)

class ImprovedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.1):
        super(ImprovedContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, output1, output2, label):
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Separate positive and negative pairs
        pos_mask = (label == 1).squeeze()
        neg_mask = (label == 0).squeeze()
        
        loss = torch.tensor(0.0, device=output1.device)
        
        # Positive pairs: minimize distance
        if pos_mask.any():
            pos_dist = euclidean_distance[pos_mask]
            pos_loss = torch.mean(pos_dist ** 2)
            loss += pos_loss
        
        # Negative pairs: maximize distance (with margin)
        if neg_mask.any():
            neg_dist = euclidean_distance[neg_mask]
            neg_loss = torch.mean(F.relu(self.margin - neg_dist) ** 2)
            loss += neg_loss
            
        return loss

class BalancedFaceDataset(Dataset):
    def __init__(self, target_person_folder, other_people_folders, transform=None, pairs_per_epoch=2000):
        self.target_person_folder = Path(target_person_folder)
        self.other_people_folders = [Path(folder) for folder in other_people_folders] if other_people_folders else []
        self.transform = transform
        self.pairs_per_epoch = pairs_per_epoch
        
        # Load target person images
        self.target_images = self._load_images(self.target_person_folder)
        if len(self.target_images) < 2:
            raise ValueError(f"Need at least 2 images of target person, found {len(self.target_images)}")
        
        # Load other people's images
        self.other_images = []
        for folder in self.other_people_folders:
            self.other_images.extend(self._load_images(folder))
        
        print(f"Loaded {len(self.target_images)} target person images")
        print(f"Loaded {len(self.other_images)} other people images")
        
        # If no other people provided, create synthetic negatives using augmentation
        self.use_synthetic_negatives = len(self.other_images) == 0
        if self.use_synthetic_negatives:
            print("No other people images provided. Using heavily augmented images as negatives.")

    def _load_images(self, folder):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(list(folder.glob(ext)))
            images.extend(list(folder.glob(ext.upper())))
        return images

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        # 50% positive pairs, 50% negative pairs
        if random.random() < 0.5:
            return self._get_positive_pair()
        else:
            return self._get_negative_pair()

    def _get_positive_pair(self):
        # Sample two different images of the target person
        img_paths = random.sample(self.target_images, 2)
        img1 = self._load_and_transform_image(img_paths[0])
        img2 = self._load_and_transform_image(img_paths[1])
        return img1, img2, torch.tensor([1.0])

    def _get_negative_pair(self):
        # One image from target person, one from others
        target_img_path = random.choice(self.target_images)
        target_img = self._load_and_transform_image(target_img_path)
        
        if self.use_synthetic_negatives:
            # Use heavily augmented version of target person as negative
            synthetic_img = self._create_synthetic_negative(target_img_path)
            return target_img, synthetic_img, torch.tensor([0.0])
        else:
            # Use image from different person
            other_img_path = random.choice(self.other_images)
            other_img = self._load_and_transform_image(other_img_path)
            return target_img, other_img, torch.tensor([0.0])

    def _create_synthetic_negative(self, img_path):
        """Create a heavily augmented version as synthetic negative"""
        synthetic_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.3),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.8),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(img_path).convert('RGB')
        return synthetic_transform(img)

    def _load_and_transform_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def validate_model(model, device, val_loader, criterion):
    """Validation function to monitor training progress"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            
            loss = criterion(output1, output2, label)
            total_loss += loss.item()
            
            # Calculate accuracy based on distance threshold
            distances = F.pairwise_distance(output1, output2)
            predictions = (distances < 0.8).float()  # Threshold for same person
            correct_predictions += (predictions == label.squeeze()).sum().item()
            total_predictions += label.size(0)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return total_loss / len(val_loader), accuracy

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    positive_losses = []
    negative_losses = []
    
    for batch_idx, (img1, img2, label) in enumerate(train_loader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Track positive vs negative losses
        pos_mask = (label == 1).squeeze()
        neg_mask = (label == 0).squeeze()
        
        if pos_mask.any():
            pos_loss = F.pairwise_distance(output1[pos_mask], output2[pos_mask]).mean()
            positive_losses.append(pos_loss.item())
        
        if neg_mask.any():
            neg_loss = F.pairwise_distance(output1[neg_mask], output2[neg_mask]).mean()
            negative_losses.append(neg_loss.item())
        
        if batch_idx % 10 == 0:
            pos_avg = np.mean(positive_losses[-10:]) if positive_losses else 0
            neg_avg = np.mean(negative_losses[-10:]) if negative_losses else 0
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Pos Dist: {pos_avg:.4f}, Neg Dist: {neg_avg:.4f}')
    
    return total_loss / len(train_loader)

def create_validation_split(target_folder, other_folders, val_ratio=0.2):
    """Create validation dataset from a portion of training data"""
    target_images = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    for ext in extensions:
        target_images.extend(list(Path(target_folder).glob(ext)))
        target_images.extend(list(Path(target_folder).glob(ext.upper())))
    
    # Split target images
    val_size = max(1, int(len(target_images) * val_ratio))
    val_target = random.sample(target_images, val_size)
    
    # Create temporary validation folders
    val_target_folder = Path("temp_val_target")
    val_target_folder.mkdir(exist_ok=True)
    
    for img_path in val_target:
        import shutil
        shutil.copy2(img_path, val_target_folder)
    
    return str(val_target_folder), other_folders

def main():
    parser = argparse.ArgumentParser(description='Improved transfer learning for face recognition')
    parser.add_argument('person_name', type=str, help='Name of the target person')
    parser.add_argument('--model', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--target-images', type=str, required=True, help='Path to target person images')
    parser.add_argument('--other-images', type=str, nargs='*', default=[], 
                       help='Paths to other people\'s image folders')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Input batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--pairs-per-epoch', type=int, default=2000, help='Number of pairs per epoch')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--validate', action='store_true', help='Enable validation')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Improved data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ], p=0.7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = BalancedFaceDataset(
        target_person_folder=args.target_images,
        other_people_folders=args.other_images,
        transform=train_transform,
        pairs_per_epoch=args.pairs_per_epoch
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)

    # Optional validation
    val_loader = None
    if args.validate and len(args.other_images) > 0:
        val_target_folder, val_other_folders = create_validation_split(
            args.target_images, args.other_images
        )
        val_dataset = BalancedFaceDataset(
            target_person_folder=val_target_folder,
            other_people_folders=val_other_folders,
            transform=val_transform,
            pairs_per_epoch=200
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=2)

    # Model setup
    model = SiameseNetwork(embedding_dim=args.embedding_dim).to(device)
    
    if os.path.exists(args.model):
        print(f"Loading pretrained model from {args.model}")
        model.load_state_dict(torch.load(args.model, map_location=device))
    else:
        print("No pretrained model found, training from scratch")

    # Training setup with different learning rates for different parts
    facenet_params = list(model.facenet.parameters())
    additional_params = list(model.additional.parameters())
    
    optimizer = optim.AdamW([
        {'params': [p for p in facenet_params if p.requires_grad], 'lr': args.lr * 0.1},  # Lower LR for pretrained
        {'params': additional_params, 'lr': args.lr}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = ImprovedContrastiveLoss(margin=1.0)

    best_loss = float('inf')
    best_accuracy = 0.0
    patience = 5
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Target person: {args.person_name}")
    print(f"Using {'real' if args.other_images else 'synthetic'} negative examples")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Training
        avg_train_loss = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        scheduler.step()
        
        # Validation
        if val_loader:
            avg_val_loss, val_accuracy = validate_model(model, device, val_loader, criterion)
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}")
            
            # Save best model based on validation accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                model_path = f"best_{args.person_name.lower().replace(' ', '_')}_model.pth"
                torch.save(model.state_dict(), model_path)
                print(f"✓ Saved best model (Val Acc: {best_accuracy:.4f})")
            else:
                patience_counter += 1
        else:
            print(f"Train Loss: {avg_train_loss:.4f}")
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                patience_counter = 0
                model_path = f"best_{args.person_name.lower().replace(' ', '_')}_model.pth"
                torch.save(model.state_dict(), model_path)
                print(f"✓ Saved best model (Loss: {best_loss:.4f})")
            else:
                patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch} epochs (patience: {patience})")
            break
    
    print(f"\nTraining completed!")
    print(f"Best model saved as: {model_path}")
    
    # Cleanup validation folders
    if args.validate:
        import shutil
        try:
            shutil.rmtree("temp_val_target")
        except:
            pass

if __name__ == '__main__':
    main()
