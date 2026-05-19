# test_standard_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import squeezenet1_1

# 使用 torchvision 自带的 SqueezeNet
device = torch.device("cpu")
model = squeezenet1_1(pretrained=False, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 假数据
x = torch.randn(100, 3, 224, 224)
y = torch.tensor([0, 1] * 50)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("测试 torchvision 自带的 SqueezeNet...")
for epoch in range(30):
    model.train()
    total_loss = 0
    correct = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch_y).sum().item()

    if (epoch + 1) % 5 == 0:
        acc = correct / len(dataset)
        print(f"Epoch {epoch + 1}: Loss={total_loss / len(loader):.4f}, Acc={acc:.2%}")