# CIFAR-10 Image Classifier (CNN) 🚀

這是一個基於 PyTorch 開發的彩色圖像辨識專案，使用 **CIFAR-10** 資料集進行訓練，並支援自定義圖片預測。

## 📌 專案亮點
* **模型架構**：自定義三層卷積神經網路 (CNN)。
* **硬體加速**：支援 Mac **MPS (Metal Performance Shaders)** 加速。
* **資料增強**：整合隨機翻轉、旋轉與顏色抖動，提升模型強健性。
* **實用工具**：包含完整訓練腳本及單張圖片預測腳本。

## 📂 專案結構
```text
CIFAR-10-AI/
├── .venv/                # 虛擬環境 (Git 已忽略)
├── data/                 # CIFAR-10 原始資料集 (Git 已忽略)
├── .gitignore            # Git 忽略設定
├── cifar10_torch.py      # 主訓練腳本 (包含資料增強邏輯)
├── predict.py            # 單張圖片預測工具
├── cifar10_model.pth     # 訓練好的模型權重 (需手動訓練產生)
└── README.md             # 專案說明文件