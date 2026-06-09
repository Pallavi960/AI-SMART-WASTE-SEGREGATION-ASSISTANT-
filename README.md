# ♻️ AI Smart Waste Segregation Assistant
### SDG 12 — Responsible Consumption & Production | 1M1B Internship

---

## 📁 Project Structure

```
waste_segregation/
├── phase1_check_dataset.py     ← Verify your dataset is ready
├── phase2_preprocessing.py     ← Image preprocessing + augmentation
├── phase3_build_model.py       ← CNN model architecture
├── phase4_train_model.py       ← Full training pipeline
├── phase5_evaluate_model.py    ← Accuracy + confusion matrix
├── phase6_test_model.py        ← Test on a single image
├── phase7_app.py               ← Streamlit web app
├── requirements.txt            ← Python dependencies
└── dataset/                    ← (YOU MUST ADD THIS)
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

---

## 🚀 Step-by-Step Run Guide

### Step 0 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 1 — Download Dataset
1. Go to https://www.kaggle.com/
2. Search: **TrashNet dataset**
3. Download + extract ZIP
4. Rename folder to `dataset`
5. Place inside this project folder

### Step 2 — Verify Dataset
```bash
python phase1_check_dataset.py
```
Expected output: ✅ all 6 class folders with images

### Step 3 — Preview Preprocessed Images (optional)
```bash
python phase2_preprocessing.py
```

### Step 4 — Build Model (optional standalone check)
```bash
python phase3_build_model.py
```

### Step 5 — TRAIN THE MODEL ⭐
```bash
python phase4_train_model.py
```
- Training time: ~10–30 min (laptop) or ~5–10 min (Google Colab)
- Saves best model as `waste_model.h5`
- Generates training accuracy/loss charts

### Step 6 — Evaluate Model
```bash
python phase5_evaluate_model.py
```
- Shows overall accuracy
- Generates confusion matrix
- Per-class accuracy bar chart

### Step 7 — Test on a Single Image
```bash
python phase6_test_model.py path/to/your/image.jpg
```
Example:
```bash
python phase6_test_model.py test_bottle.jpg
```

### Step 8 — Launch Web App 🌐
```bash
streamlit run phase7_app.py
```
Opens at http://localhost:8501

---

## 🧠 Model Architecture
- **Base:** MobileNetV2 (pre-trained on ImageNet)
- **Training Strategy:** Transfer learning + fine-tuning
- **Classes:** cardboard, glass, metal, paper, plastic, trash
- **Expected Accuracy:** 88–95%

## ⚖️ Responsible AI Statement
- Model may be incorrect for unclear or unusual images
- Designed for awareness, not enforcement
- Dataset may have geographic bias (US-based photos)
- No personal data is collected or stored
- Always follow your local recycling guidelines

---

*Built for 1M1B SDG Internship | SDG 12: Responsible Consumption & Production*
