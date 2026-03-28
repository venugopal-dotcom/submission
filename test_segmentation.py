import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# =============================
# Dice Loss (Binary) - Copied from training script
# =============================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        preds = preds[:, 1, :, :]
        targets = targets.float()

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

# =============================
# IoU Metric - Copied from training script (though might not be used directly in test, good to have)
# =============================
def compute_iou(preds, targets):
    preds = torch.argmax(preds, dim=1)
    intersection = ((preds == 1) & (targets == 1)).sum().float()
    union = ((preds == 1) | (targets == 1)).sum().float()
    if union == 0:
        return torch.tensor(1.0).to(preds.device)
    return intersection / union

# =============================
# Strong Decoder (UNet-style) - Copied from training script
# =============================
class Decoder(nn.Module):
    def __init__(self, in_channels=384, num_classes=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )
    def forward(self, x):
        return self.block(x)

# =============================
# Full Model - Copied from training script
# =============================
class DinoSegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vits14'
        )
        self.backbone.head = nn.Identity()
        self.decoder = Decoder()

    def forward(self, x):
        features = self.backbone.forward_features(x)["x_norm_patchtokens"]
        B, N, C = features.shape
        H = W = int(N ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.decoder(features)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out

# =============================
# Test Dataset
# =============================
class TestDataset(Dataset):
    def __init__(self, image_dir, image_size=392):
        self.image_dir = image_dir
        self.image_size = image_size
        self.images = sorted(os.listdir(image_dir))

        self.transform_image = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform_image(image)
        return image, img_name

# =============================
# MAIN
# =============================
def main():
    parser = argparse.ArgumentParser(description="Test segmentation model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--test_data_dir", type=str, default="/content/drive/MyDrive/Offroad_Segmentation/Offroad_Segmentation_testImages/Color_Images", help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/Offroad_Segmentation/Predictions", help="Directory to save predicted masks")
    parser.add_argument("--image_size", type=int, default=392, help="Input image size (must be divisible by 14 for DINOv2_vits14)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Prepare dataset and dataloader
    print("Loading dataset from", args.test_data_dir, "...")
    test_dataset = TestDataset(args.test_data_dir, image_size=args.image_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Loaded", len(test_dataset), "samples")

    # Load model
    print("Loading DINOv2 backbone...")
    model = DinoSegModel().to(device) # Correctly instantiate DinoSegModel
    print("Backbone loaded successfully!")

    print("Loading model from", args.model_path, "...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully!")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print("Saving predictions to", args.output_dir)

    # Inference
    for images, img_names in tqdm(test_loader):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for i, pred_mask in enumerate(preds):
            output_mask_image = Image.fromarray((pred_mask * 255).astype(np.uint8))
            output_mask_image = output_mask_image.resize(
                (test_dataset.image_size, test_dataset.image_size),
                Image.NEAREST
            )
            output_path = os.path.join(args.output_dir, img_names[i])
            output_mask_image.save(output_path)

    print("Testing complete. Predicted masks saved to", args.output_dir)

if __name__ == "__main__":
    main()
