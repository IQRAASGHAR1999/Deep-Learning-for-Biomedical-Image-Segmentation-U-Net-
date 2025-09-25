import torch
from torch.utils.data import DataLoader
from src.unet import UNet
from src.data_loader import SegmentationDataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import os

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = SegmentationDataset('data/images', 'data/masks', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(2):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item()}")
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/unet.pth")
