import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================== Hyperparameters ==========================
num_classes = 2           # Defective / Non defective
batch_size = 16           # smaller batch for small dataset
num_channels = 3          # RGB images
img_size = 224            # resize railway images to 224x224
patch_size = 16           # 224 / 16 = 14 patches per side
embedding_dim = 128
attention_heads = 4
transformer_blocks = 6
mlp_hidden_nodes = 256
learning_rate = 3e-4
epochs = 30
dropout_rate = 0.1        # regularisation for small dataset

num_patches = (img_size // patch_size) ** 2   # 196 patches

# ========================== Data Paths ==========================
data_root = "archive/Railway Track fault Detection Updated"

# ========================== Transforms ==========================
# Augmentation for training (helps with only 300 images)
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],    # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ========================== Datasets & Loaders ==========================
train_dataset = ImageFolder(root=f"{data_root}/Train", transform=train_transform)
val_dataset   = ImageFolder(root=f"{data_root}/Validation", transform=eval_transform)
test_dataset  = ImageFolder(root=f"{data_root}/Test", transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

class_names = train_dataset.classes   # ['Defective', 'Non defective']
print(f"Classes: {class_names}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ========================== Model Components ==========================

class PatchEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            num_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.patch_embed(x)            # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)                   # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)              # (B, num_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.attn = nn.MultiheadAttention(
            embedding_dim,
            attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_nodes),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_nodes, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # Self-attention with residual
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + res

        # MLP with residual
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + res

        return x


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_embedding = PatchEmbeddings()

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embedding_dim)
        )
        self.pos_drop = nn.Dropout(dropout_rate)

        self.transformer = nn.Sequential(
            *[TransformerEncoder() for _ in range(transformer_blocks)]
        )

        self.mlp_head = MLPHead()

    def forward(self, x):
        x = self.patch_embedding(x)

        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.pos_drop(x)

        x = self.transformer(x)

        x = x[:, 0]       # CLS token
        x = self.mlp_head(x)

        return x


# ========================== Training Setup ==========================
model = VisionTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ========================== Training & Validation Loop ==========================
best_val_acc = 0.0

for epoch in range(epochs):
    # --- Train ---
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # --- Validation ---
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}  |  "
          f"Train Loss: {total_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_railway_vit.pth")
        print(f"  >> Saved best model (Val Acc: {val_acc:.2f}%)")

# ========================== Test Evaluation ==========================
print("\n" + "=" * 60)
print("TESTING on held-out test set")
print("=" * 60)

model.load_state_dict(torch.load("best_railway_vit.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = 100 * (all_preds == all_labels).sum() / len(all_labels)
print(f"\nTest Accuracy: {test_acc:.2f}%")
print(f"\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
print(f"Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
