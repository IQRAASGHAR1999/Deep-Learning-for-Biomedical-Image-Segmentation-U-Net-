import torch
from src.unet import UNet
from src.data_loader import SegmentationDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_dataset = SegmentationDataset('data/images', 'data/masks', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("results/unet.pth", map_location=device))
model.eval()

def dice(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    intersection = (pred & (target > 0.5)).float().sum()
    return (2. * intersection) / (pred.float().sum() + target.float().sum() + eps)

total_dice = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        total_dice += dice(outputs, masks).item()
print(f"Mean Dice: {total_dice / len(test_loader):.4f}")
