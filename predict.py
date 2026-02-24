import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('MacOSX') # 這是 Mac 專用的原生後端
import matplotlib.pyplot as plt
import os

# --- 1. 定義結構 (必須與訓練時的 CIFARNet 完全一致) ---
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

# 10 個類別名稱
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def predict_custom_image(image_path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 載入模型
    model = CIFARNet().to(device)
    MODEL_PATH = "cifar10_model.pth"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型檔 {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # --- 2. 預處理彩色圖片 ---
    transform = transforms.Compose([
        transforms.Resize((32, 32)),     # 強制縮放成 32x32
        transforms.ToTensor(),           # 轉為張量 (0-1)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 與訓練時相同的正規化
    ])

    try:
        img_original = Image.open(image_path).convert('RGB') # 確保是 RGB 三通道
        img_tensor = transform(img_original).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ 讀取圖片失敗: {e}")
        return

    # --- 3. 執行預測 ---
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        class_idx = predicted.item()
        conf_score = confidence.item() * 100

    print(f"🔍 預測結果: {classes[class_idx]} (信心指數: {conf_score:.2f}%)")

    # --- 4. 顯示結果 ---
    plt.imshow(img_original)
    plt.title(f"Prediction: {classes[class_idx]} ({conf_score:.2f}%)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 在這裡填入你的圖片檔名
    test_image = "my_dog.jpg" 
    if os.path.exists(test_image):
        predict_custom_image(test_image)
    else:
        print(f"請放入一張圖片並命名為 {test_image}，或修改程式碼中的路徑。")