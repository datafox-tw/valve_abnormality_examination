# 🫀 3D Cardiac Image Segmentation with Self-Supervised Learning (SSL)
## AI Cup 2025: Heart Muscle Image Segmentation - Top 10% (Rank 56/568)

[繁體中文版](./readme-chinese.md)

[![AI Cup Rank](https://img.shields.io/badge/AI_Cup-Rank_56/568-blue.svg)](https://tbrain.trendmicro.com.tw/Competitions/Details/39)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

> This project implements a cutting-edge 3D medical image segmentation pipeline for cardiac anatomy, leveraging **Self-Supervised Learning (SSL)** to achieve state-of-the-art performance. The approach was validated in the **AI Cup 2025**, securing a position in the **Top 10%** among 568 competing teams.

---

## 📺 Project Presentation
Watch our detailed project walkthrough and methodology presentation here:

[![Watch the video](https://img.youtube.com/vi/yOSHRBcBeeA/0.jpg)](https://www.youtube.com/watch?v=yOSHRBcBeeA)

---

## 🌟 Highlights
- **Performance**: Outperformed the strong nnU-Net baseline by ~3 Dice points.
- **Methodology**: Advanced Masked Auto Encoder (MAE) pre-training for 3D CNNs.
- **Architecture**: Residual Encoder U-Net within the nnU-Net framework.
- **Dataset**: Pre-trained on 39,000 Brain MRI volumes; Fine-tuned on Cardiac CT/MRI.

---

## 📖 From Research to Replication (論文發想到復現)

### Inspiration (論文啟發)
Our work is heavily inspired by the CVPR 2025 paper:
**"Revisiting MAE Pre-training for 3D Medical Image Segmentation"** (Wald et al.).

The paper addresses three critical pitfalls in 3D medical SSL:
1. **Small pre-training datasets**: Most studies use limited data.
2. **Inadequate architectures**: Standard 2D or basic 3D CNNs often underperform.
3. **Insufficient evaluation**: Lack of rigorous benchmarking.

### Our Implementation Path (復現與應用流程)

#### 1. Massive Pre-training (大規模預訓練)
We leveraged a massive dataset of **39,000 3D Brain MRI volumes**. By using **Masked Auto Encoders (MAEs)**, the model learned high-level anatomical features by reconstructing missing parts of the 3D volume. This "self-supervised" phase allows the model to understand 3D spatial relationships without requiring expensive manual labels.

我們利用了 **39,000 個 3D 腦部 MRI 影像** 進行大規模預訓練。透過 **Masked Auto Encoders (MAEs)** 技術，模型學會了從殘缺的影像中重建出完整的 3D 結構，從而掌握了深層的解剖學特徵。這種自監督學習（SSL）在大規模未標註數據上展現了極大的潛力。

#### 2. Residual Encoder U-Net (架構優化)
Instead of a standard U-Net, we integrated a **Residual Encoder U-Net** into the **nnU-Net** framework. This combines the robustness of nnU-Net's automated preprocessing with the powerful feature extraction of residual connections, specifically tailored for 3D medical image analysis.

我們在 **nnU-Net** 框架中採用了 **Residual Encoder U-Net** 架構。這結合了 nnU-Net 強大的自動化預處理流程與殘差連接（Residual Connections）的特徵提取能力，專為 3D 醫療影像分割進行了優化。

#### 3. Real-World Application: AI Cup (真實應用套用)
We transferred the pre-trained weights to the task of **Heart Muscle Segmentation (Myocardium, LA, LV)** for the AI Cup 2025. 
- **Result**: Our model achieved a **Top 10%** ranking (**56/568**).
- **Impact**: The SSL approach provided a **~3 Dice point improvement** over the standard supervised baseline, demonstrating that pre-training on brain data significantly benefits cardiac segmentation.

我們將預訓練好的權重遷移至 **AI Cup 2025 心臟肌肉影像分割** 任務中。最終在 568 支隊伍中脫穎而出，獲得 **第 56 名（前 10%）** 的佳績。實驗證明，相較於純監督式學習，自監督預訓練讓 Dice 系數提升了約 **3 個百分點**。

---

## 🛠 Project Components (專案組成)

- **`MAE.py`**: Self-supervised pre-training using the `nnssl` framework.
- **`FT.py`**: Task-specific fine-tuning on AI Cup cardiac data.
- **`analyze_dataset.py` & `visualize_data.py`**: Quality assurance and 3D visualization tools.
- **Reproduction Guide**: Detailed steps on environment setup and code patches: [REPRODUCTION.md](./REPRODUCTION.md)

---

## 🚀 Reproduction Quick Start (核心執行流程)

For a step-by-step guide to installing dependencies and applying necessary patches, please refer to our **[Reproduction Guide](./REPRODUCTION.md)**.

1. **Pre-training**: `python3 MAE.py`
2. **Fine-tuning**: `python3 FT.py`
3. **Validation**: `python3 visualize_data.py`

---

## 📜 Acknowledgements & References
- T. Wald et al., "Revisiting MAE Pre-training for 3D Medical Image Segmentation," *CVPR*, 2025.
- F. Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods*, 2021.
- AI Cup 2025 Organizers for providing the dataset and platform.

---

## 📁 Repository Structure
```
.
├── 📄 README.md                 # Main showcase
├── 📄 REPRODUCTION.md           # Step-by-step setup and patches
├── 📄 MAE.py                    # Pre-training script
├── 📄 FT.py                     # Fine-tuning script
├── 📄 analyze_dataset.py        # Data analysis tools
├── 📄 visualize_data.py         # 3D/Slice visualization
├── 📄 pipeline_generation.py    # Pipeline config generation
└── 📁 ...                       # PDFs and other assets
```
