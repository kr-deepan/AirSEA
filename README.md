# 🌊 Oil Spill Detection (Phase 1: Classification)

This project demonstrates the detection of **oil spills in oceans** using **Sentinel-1 SAR satellite images** and **deep learning (ResNet18)**.  
It leverages the [CSIRO Marine Oil Spill Dataset](https://data.csiro.au/collection/csiro:57430), which contains labeled SAR image chips (oil vs. no oil).  

---

## 🚀 Project Overview
- Train a **ResNet18 classifier** to distinguish between:
  - **No Oil Spill (class 0)**
  - **Oil Spill Present (class 1)**
- Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix.
- Deploy a simple **Streamlit demo app** to test predictions interactively.

---

## 📂 Project Structure
```
oil_spill_detection/
├── data/                         # Dataset directory
│   ├── class_0/                  # No oil images
│   ├── class_1/                  # Oil spill images
│   └── labels.csv                # Filename → label mapping
│
├── models/                       # Trained model weights
│   └── resnet18_baseline.pth
│
├── notebooks/                    # Jupyter notebook for training
│   └── phase1_classification.ipynb
│
├── utils/                        # Helper scripts
│   └── dataloader.py
│
├── outputs/                      # Logs, reports
│   └── classification_report.csv
│
├── app/                          # Streamlit demo app
│   └── demo_app.py
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Installation
1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/oil_spill_detection.git
   cd oil_spill_detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv oil_spill_detection_env
   .\oil_spill_detection_env\Scripts\activate   # Windows
   # OR
   source oil_spill_detection_env/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Training
Run the training notebook:
```
notebooks/phase1_classification.ipynb
```

During training:
- Model is saved to `models/resnet18_baseline.pth`
- Metrics and classification report are stored in `outputs/`

---

## 📊 Example Results
| Metric      | Score |
|-------------|-------|
| Accuracy    | 98%   |
| Precision   | 0.97  |
| Recall      | 0.97  |
| F1-Score    | 0.97  |

Confusion Matrix:
```
[[739  11]
 [ 11 365]]
```

---

## 🖥️ Demo App
Run the Streamlit app:
```bash
streamlit run app/demo_app.py
```

Upload a **400×400 SAR chip** and the model will classify it as:
- 🛢️ **Oil Spill Detected**
- 🌊 **No Oil Spill**

---

## 📚 Dataset Credit
This project uses the **Marine Oil Spill Dataset** provided by **CSIRO**:  
🔗 [https://data.csiro.au/collection/csiro:57430](https://data.csiro.au/collection/csiro:57430)

---

## 🙌 Acknowledgements
- [PyTorch](https://pytorch.org/) for deep learning framework  
- [Torchvision](https://pytorch.org/vision/stable/index.html) for pretrained models  
- [Streamlit](https://streamlit.io/) for interactive demo  
- **CSIRO** for curating and publishing the Sentinel-1 SAR oil spill dataset  
