import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('MacOSX') # 這是 Mac 專用的原生後端
import matplotlib.pyplot as plt
import time
import os

# --- 1. 設定設備 ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 使用設備: {device}")

# CIFAR-10 的 10 個類別
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- 2. 資料預處理與增強 ---
# 彩色圖片通常需要比較強的正規化
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 隨機左右翻轉
    transforms.RandomRotation(10),     # 隨機旋轉 10 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 隨機調整亮度對比
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# --- 3. 定義稍微強大一點的 CNN ---
class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        # 增加通道數：3 -> 64 -> 128 -> 256
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 注意：這裡的維度會改變，如果是 3 次 pool，32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.3) # 稍微提高 Dropout 防止過擬合

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CIFARNet().to(device)
MODEL_PATH = "cifar10_model.pth"

# --- 4. 訓練邏輯 ---
if os.path.exists(MODEL_PATH):
    choice = input("偵測到模型存檔，是否載入？ (y/n): ").lower()
else:
    choice = 'n'

if choice == 'y':
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ 已載入現有模型")
else:
    print("🔥 開始訓練 (預計 10 輪，彩色辨識較難)...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 17):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch} | Loss: {running_loss/len(train_loader):.4f} | Time: {time.time()-start_time:.2f}s")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("💾 訓練完成並存檔")

# --- 5. 隨堂測驗 ---
model.eval()
data_iter = iter(test_loader)
images, labels = next(data_iter)

with torch.no_grad():
    output = model(images.to(device))
    _, predicted = torch.max(output, 1)

# 顯示前 4 張圖
fig = plt.figure(figsize=(10, 4))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1)
    # 反正規化以正常顯示圖片
    img = images[i] / 2 + 0.5
    plt.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(f"Pred: {classes[predicted[i]]}\nReal: {classes[labels[i]]}")
    plt.axis('off')
plt.show()