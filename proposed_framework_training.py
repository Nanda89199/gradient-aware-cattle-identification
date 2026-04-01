# =========================================================
# CATTLE IDENTIFICATION - FINAL STABLE VERSION
# EfficientNetV2 + SE + Gradient + Multi-Scale Fusion
# =========================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===============================
# CUDA SAFETY
# ===============================
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()

# ===============================
# CONFIG
# ===============================
data_dir = r'/kaggle/working/cattle_t_v_t'
batch_size = 8
epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# CLEAN EMPTY FOLDERS
# ===============================
for split in ["train", "val", "test"]:
    split_path = os.path.join(data_dir, split)
    for root, dirs, files in os.walk(split_path):
        if len(files) == 0 and root != split_path:
            os.rmdir(root)

# ===============================
# TRANSFORMS
# ===============================
train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===============================
# DATASETS
# ===============================
train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)
test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_tfms)

val_ds.class_to_idx = train_ds.class_to_idx
test_ds.class_to_idx = train_ds.class_to_idx

num_classes = len(train_ds.classes)
print(f"Detected number of classes: {num_classes}")

# ===============================
# LOADERS
# ===============================
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ===============================
# SE BLOCK
# ===============================
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // reduction)
        self.fc2 = nn.Linear(ch // reduction, ch)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# ===============================
# GRADIENT BLOCK
# ===============================
class GradientBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)

        sx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        sy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
        self.register_buffer("sx", sx.view(1,1,3,3))
        self.register_buffer("sy", sy.view(1,1,3,3))

    def forward(self, x):
        gray = x.mean(1, keepdim=True)
        gx = F.conv2d(gray, self.sx, padding=1)
        gy = F.conv2d(gray, self.sy, padding=1)
        g = torch.sqrt(gx**2 + gy**2 + 1e-6)
        g = g.repeat(1, x.size(1), 1, 1)
        return F.relu(self.bn(self.dw(g)), inplace=True)

# ===============================
# MULTI-SCALE FUSION
# ===============================
class MultiScaleFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch*2, ch, 1)
        self.c3 = nn.Conv2d(ch*2, ch, 3, padding=1)
        self.c5 = nn.Conv2d(ch*2, ch, 5, padding=2)
        self.bn = nn.BatchNorm2d(ch*3)

    def forward(self, a, b):
        x = torch.cat([a, b], 1)
        f1 = self.c1(x)
        f3 = self.c3(x)
        f5 = self.c5(x)
        return F.relu(self.bn(torch.cat([f1,f3,f5], 1)), inplace=True)

# ===============================
# MODEL (UPDATED TO features[6])
# ===============================
class CattleNet(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        base = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.features = base.features

        ch = 256  # 🔴 features[6] output channels

        self.se = SEBlock(ch)
        self.grad = GradientBlock(ch)
        self.fuse = MultiScaleFusion(ch)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch*3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, ncls)
        )

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 6:   # 🔴 STOP at features[6]
                break

        x1 = self.se(x)
        x2 = self.grad(x)
        x = self.fuse(x1, x2)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# ===============================
# TRAIN SETUP
# ===============================
model = CattleNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler("cuda")

# ===============================
# TRAIN LOOP
# ===============================
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = 100 * train_correct / train_total

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
    )

# ===============================
# FINAL TEST
# ===============================
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        preds = model(imgs).argmax(1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = 100 * test_correct / test_total
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

torch.save(model.state_dict(), "cattlenet_final.pth")
print("Model saved successfully.")
