# =========================================================
# CREATE EMBEDDINGS USING TRAINED CattleNet MODEL
# =========================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np

# ===============================
# SETTINGS
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_path = "/kaggle/working/Feature6_cattle1_Novel_final.pth"   # your trained model
pairs_csv_path = "/kaggle/working/pairs_test.csv"    
images_base_path = "/kaggle/working/cattle_t_v_t"  # dataset base
embeddings_save_path = "/kaggle/working/cattle1_Grams_test_embeddings.npy"  # where to save embeddings

# Image transform (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===============================
# MODEL DEFINITIONS (CattleNet)
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

class CattleNet(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        base = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.features = base.features
        ch = 256  # features[6] output channels
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

    def forward(self, x, return_embedding=False):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 6:
                break
        x1 = self.se(x)
        x2 = self.grad(x)
        x = self.fuse(x1, x2)
        x = self.pool(x).flatten(1)
        if return_embedding:
            return x  # return embedding instead of logits
        return self.fc(x)

# ===============================
# LOAD MODEL
# ===============================
# first, detect number of classes from folder
num_classes = len([d for d in os.listdir(os.path.join(images_base_path, "train")) if os.path.isdir(os.path.join(images_base_path, "train", d))])
model = CattleNet(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Loaded model with {num_classes} classes")

# ===============================
# LOAD CSV
# ===============================
df = pd.read_csv(pairs_csv_path)
all_images = set(df['img1'].tolist() + df['img2'].tolist())
print(f"Total unique images in pairs: {len(all_images)}")

# ===============================
# GENERATE EMBEDDINGS
# ===============================
embeddings = {}
with torch.no_grad():
    for img_path in all_images:
        # full path
        full_path = os.path.join(images_base_path, img_path)
        if not os.path.exists(full_path):
            print(f"⚠ Missing: {full_path}")
            continue
        img = Image.open(full_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        emb = model(img_tensor, return_embedding=True)  # get embedding
        embeddings[img_path] = emb.cpu().numpy()

# ===============================
# SAVE EMBEDDINGS
# ===============================
np.save(embeddings_save_path, embeddings)
print(f"✅ Embeddings saved to {embeddings_save_path}")
