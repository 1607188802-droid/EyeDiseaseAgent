import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd


# ========== 数据集类 ==========
class EyeDataset(Dataset):
    def __init__(self, image_dir, label_file=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []

        if label_file and os.path.exists(label_file):
            df = pd.read_excel(label_file)
            label_map = dict(zip(df['imgName'], df['Label']))
            for img_name in os.listdir(image_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    if img_name in label_map:
                        self.images.append(img_name)
                        self.labels.append(label_map[img_name])
        else:
            for filename in os.listdir(image_dir):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(filename)
                    self.labels.append(1 if filename.upper().startswith('P') else 0)

        print(f"📁 {os.path.basename(image_dir)}: {len(self.images)}张, 病理性{sum(self.labels)}张")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ========== 数据预处理 ==========
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 加载数据 ==========
train_dir = r"D:\SqueezeNet\data\PALM-Training400\PALM-Training400"
val_dir = r"D:\SqueezeNet\data\PALM-Validation400"
label_file = r"D:\SqueezeNet\data\PALM-Validation-GT\PM_Label_and_Fovea_Location.xlsx"

train_dataset = EyeDataset(train_dir, transform=train_transform)
val_dataset = EyeDataset(val_dir, label_file=label_file, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ========== 使用 ResNet18（预训练） ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n💻 使用设备: {device}")

# 加载预训练的 ResNet18
model = models.resnet18(pretrained=True)
# 替换最后一层为二分类
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

# ========== 训练 ==========
print("\n🚀 开始训练 ResNet18...\n")
best_acc = 0

for epoch in range(30):
    # 训练
    model.train()
    train_correct = 0
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset)

    # 验证
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)

    print(f"Epoch {epoch + 1:2d}: Loss={train_loss / len(train_loader):.4f}, "
          f"Train Acc={train_acc:.2%}, Val Acc={val_acc:.2%}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_resnet18.pth")
        print(f"  ✅ 保存模型 (验证准确率: {best_acc:.2%})")

    scheduler.step(val_acc)

print(f"\n🎉 训练完成！最佳验证准确率: {best_acc:.2%}")