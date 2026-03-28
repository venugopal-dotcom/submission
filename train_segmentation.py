import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# =============================
# Dice Loss (Binary)
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
# IoU Metric
# =============================

def compute_iou(preds, targets):
    preds = torch.argmax(preds, dim=1)

    intersection = ((preds == 1) & (targets == 1)).sum().float()
    union = ((preds == 1) | (targets == 1)).sum().float()

    if union == 0:
        return torch.tensor(1.0).to(preds.device)

    return intersection / union


# =============================
# Strong Decoder (UNet-style)
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
# Full Model
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
# Training
# =============================

def train(model, train_loader, val_loader, device):

    # Class weights (tune if needed)
    weights = torch.tensor([0.3, 0.7]).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=weights)
    dice_loss = DiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50
    )

    scaler = torch.cuda.amp.GradScaler()

    best_iou = 0

    # Freeze backbone first 5 epochs
    for param in model.backbone.parameters():
        param.requires_grad = False

    for epoch in range(50):

        if epoch == 5:
            print("Unfreezing backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True

        model.train()
        total_loss = 0

        for images, masks in tqdm(train_loader):

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = 0.5 * ce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                iou = compute_iou(outputs, masks)
                total_iou += iou.item()

        val_iou = total_iou / len(val_loader)

        print(f"\nEpoch {epoch+1}/50")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val IoU: {val_iou:.4f}")
        print("-" * 40)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

    print("Training Complete")
    print("Best IoU:", best_iou)


# =============================
# MAIN
# =============================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DinoSegModel().to(device)

    # Replace with your dataset classes
    from dataset import TrainDataset, ValDataset

    train_dataset = TrainDataset(image_size=384)
    val_dataset = ValDataset(image_size=384)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    train(model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()