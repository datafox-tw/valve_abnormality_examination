# 🫀 3D 心臟影像分割與自監督學習 (SSL)
## AI Cup 2025: 心臟肌肉影像分割 - 前 10% (排名 56/568)

[English Version](./README.md)

[![AI Cup Rank](https://img.shields.io/badge/AI_Cup-Rank_56/568-blue.svg)](https://tbrain.trendmicro.com.tw/Competitions/Details/39)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

> 本專案實作了一套領先的 3D 醫療影像分割流水線（Pipeline），利用 **自監督學習 (Self-Supervised Learning, SSL)** 技術在心臟解剖結構分割上取得卓越成效。此方案在 **AI Cup 2025** 競賽中獲得驗證，於 568 支參賽隊伍中位列 **前 10%**。

---

## 📺 專案成果展示
查看我們的詳細專案說明與技術發想簡報：

[![觀看影片](https://img.youtube.com/vi/yOSHRBcBeeA/0.jpg)](https://www.youtube.com/watch?v=yOSHRBcBeeA)

---

## 🌟 核心亮點
- **性能卓越**：相較強大的 nnU-Net 基準模型，Dice 系數提升約 **3 個百分點**。
- **技術前沿**：針對 3D CNN 優化的 **Masked Auto Encoder (MAE)** 預訓練技術。
- **架構優化**：在 nnU-Net 框架中整合了 **Residual Encoder U-Net**。
- **數據規模**：利用 39,000 個大腦 MRI 影像進行預訓練；並在心臟 CT/MRI 數據上進行微調（Fine-tuning）。

---

## 📖 從論文發想到應用復現

### 1. 研究啟發 (Inspiration)
專案深受 CVPR 2025 論文啟發：
**"Revisiting MAE Pre-training for 3D Medical Image Segmentation"** (Wald et al.)。

該研究指出了 3D 醫療影像 SSL 的三大痛點：
1. **預訓練數據量不足**：多數研究僅使用少量數據。
2. **架構不適配**：傳統 2D 或基礎 3D CNN 難以發揮最佳效能。
3. **評估不嚴謹**：缺乏跨數據集的廣泛驗證。

### 2. 實作路徑 (Implementation Path)

#### 大規模自監督預訓練
我們採用了包含 **39,000 個 3D 大腦 MRI 影像** 的海量數據集。透過 **Masked Auto Encoders (MAEs)** 技術，模型學會了從殘缺的影像中重建完整的 3D 結構，從而掌握了深層的解剖學特徵。這種 SSL 階段讓模型在不需要昂貴的人工標註下，就能理解複雜的 3D 空間關係。

#### 架構升級：Residual Encoder U-Net
我們捨棄了標準 U-Net，轉而在 **nnU-Net** 優秀的自動化預處理框架中，引入了 **Residual Encoder U-Net**。這結合了殘差連接（Residual Connections）強大的特徵提取能力，專門針對 3D 醫療影像分割的穩定性與準確度進行優化。

#### 真實情境應用：AI Cup 競賽
我們將預訓練好的編碼器權重轉移至 **AI Cup 2025 心臟肌肉影像分割獎** 任務中：
- **成果**：在 568 支隊伍中脫穎而出，獲得 **第 56 名（前 10%）**。
- **影響**：SSL 預訓練方法讓模型的 Dice 分數比純監督式學習提升了約 **3%**，證明了在異質數據（如大腦影像）上的預訓練能顯著增益心臟影像的分割效果。

---

## 🛠 專案組成 (Project Components)

- **`MAE.py`**：使用 `nnssl` 框架進行自監督預訓練的主程式。
- **`FT.py`**：載入預訓練權重，在 AI Cup 心臟資料集上進行微調。
- **`analyze_dataset.py` & `visualize_data.py`**：資料品質檢測與 3D 視覺化工具。
- **環境設定指南**：關於庫修補（Patches）與安裝的詳細步驟請參考：[REPRODUCTION.md](./REPRODUCTION.md)

---

## 🚀 快速開始 (Quick Start)

如需完整的安裝與環境配置說明，請參閱 **[REPRODUCTION.md](./REPRODUCTION.md)**。

1. **預訓練**：`python3 MAE.py`
2. **微調**：`python3 FT.py`
3. **驗證與視覺化**：`python3 visualize_data.py`

---

## 📜 參考文獻
- T. Wald et al., "Revisiting MAE Pre-training for 3D Medical Image Segmentation," *CVPR*, 2025.
- F. Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods*, 2021.
- AI Cup 2025 主辦單位提供的數據集與平台。

---

## 📁 目錄結構
```
.
├── 📄 README.md                 # 英文專案展示
├── 📄 readme-chinese.md         # 繁體中文專案展示
├── 📄 REPRODUCTION.md           # 安裝、設定與程式碼修補指南
├── 📄 MAE.py                    # 預訓練腳本
├── 📄 FT.py                     # 微調腳本
├── 📄 analyze_dataset.py        # 資料統計工具
├── 📄 visualize_data.py         # 3D 建模與視覺化
└── 📁 ...                       # 相關技術資產與論文 PDF
```
